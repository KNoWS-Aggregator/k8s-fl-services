"""Long-running HTTP coordinator for federated averaging sessions."""
from __future__ import annotations

import json
import logging
import os
import resource
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from common.http_client import request as send_request
from common.messages import (
    ClientSessionEnd,
    ClientSessionStart,
    EvaluationFailureMessage,
    EvaluationResultMessage,
    MessageType,
    TrainingConfig,
    TrainingFailureMessage,
    TrainingResultMessage,
)
from common.weight_io import bytes_to_weights, weights_to_bytes
from fl_model import create_initial_weights, weight_signature

from .aggregate import aggregate_evaluation_metrics, federated_average
from .client_registry import ClientRegistry, next_poll_delay
from .dispatch import dispatch_evaluation, dispatch_round

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(threadName)s - %(message)s",
)
logger = logging.getLogger(__name__)


class _HealthCheckAccessFilter(logging.Filter):
    """Drop successful probe requests from Uvicorn's access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return True
        path, status_code = str(args[2]).partition("?")[0], args[4]
        return path not in {"/healthz", "/readyz"} or int(status_code) >= 400


logging.getLogger("uvicorn.access").addFilter(_HealthCheckAccessFilter())

TRAIN_PATH = "/train"
CLIENT_SESSION_START_PATH = "/session/start"
CLIENT_SESSION_END_PATH = "/session/end"
RESULT_PATH = "/training-results"
EVALUATE_PATH = "/evaluate"
EVALUATION_RESULT_PATH = "/evaluation-results"


class SessionStartRequest(BaseModel):
    expected_rounds: int = Field(ge=1)
    min_clients: int = Field(default=1, ge=1)
    round_timeout_seconds: float = Field(default=3600, gt=0)
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)


@dataclass
class ActiveSession:
    session_id: str
    request: SessionStartRequest
    all_clients: dict[str, str]
    active_clients: dict[str, str]
    model_signature: str
    session_started_at: str = ""
    round_started_at: str | None = None
    current_round: int = 0
    global_weights: bytes = b""
    expected_clients: set[str] = field(default_factory=set)
    weights_by_client: dict[str, bytes] = field(default_factory=dict)
    metrics_by_client: dict = field(default_factory=dict)
    current_aggregated_weights: bytes = b""
    failed_clients: dict[str, str] = field(default_factory=dict)
    phase: str = "training"
    evaluation_metrics_by_client: dict = field(default_factory=dict)
    aggregated_evaluation_metrics: dict[str, Any] = field(default_factory=dict)
    round_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    round_started_perf: float = 0.0
    dispatch_finished_perf: float = 0.0
    timer: threading.Timer | None = None


_coordinator_lock = threading.RLock()
_active: ActiveSession | None = None
_registry: ClientRegistry | None = None
_stop_polling = threading.Event()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "/app/data")) / "weight-aggregation"


def _status_path() -> Path:
    return _data_dir() / "status.json"


def _global_weights_path() -> Path:
    return _data_dir() / "global-weights.npz"


def _in_progress_global_weights_path() -> Path:
    return _data_dir() / "global-weights-in-progress.npz"


def _global_metadata_path() -> Path:
    return _data_dir() / "global-weights.json"


def _round_metrics_path() -> Path:
    return _data_dir() / "round-metrics.json"


def _session_dir(session_id: str) -> Path:
    return _data_dir() / "sessions" / session_id


def _callback_url() -> str:
    base = os.getenv("AGG_PUBLIC_URL", "http://weight-aggregation:8080").rstrip("/")
    return f"{base}{RESULT_PATH}"


def _evaluation_callback_url() -> str:
    base = os.getenv("AGG_PUBLIC_URL", "http://weight-aggregation:8080").rstrip("/")
    return f"{base}{EVALUATION_RESULT_PATH}"


def _peak_ram_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _client_registry() -> ClientRegistry:
    global _registry
    if _registry is None:
        case_slice = os.getenv("CASE_SLICE", "").strip()
        if not case_slice:
            raise ValueError("Required environment variable CASE_SLICE is not set")
        _registry = ClientRegistry(case_slice)
    return _registry


def _refresh_clients() -> dict[str, list[str]]:
    changes = _client_registry().refresh()
    if changes["added"] or changes["removed"]:
        logger.info(
            "Training service registry changed: %d added, %d removed",
            len(changes["added"]),
            len(changes["removed"]),
        )
    return changes


def _is_trainable(base_url: str, own_session_id: str | None) -> bool:
    try:
        response = send_request("GET", f"{base_url.rstrip('/')}/status", timeout=10)
        response.raise_for_status()
        payload = response.json()
        prepared = (payload.get("prepared_data") or {}).get("valid") is True
        session = payload.get("session") or {}
        assigned_elsewhere = session.get("active") is True and (
            not own_session_id or session.get("session_id") != own_session_id
        )
        return prepared and not assigned_elsewhere
    except Exception:
        logger.exception("Could not determine training readiness at %s", base_url)
        return False


def _client_counts() -> tuple[int, int, set[str]]:
    urls = _client_registry().snapshot()
    with _coordinator_lock:
        own_session_id = _active.session_id if _active else None
    if not urls:
        return 0, 0, set()
    with ThreadPoolExecutor(max_workers=min(16, len(urls))) as executor:
        results = dict(
            zip(
                urls,
                executor.map(
                    lambda url: _is_trainable(url, own_session_id),
                    urls,
                ),
            )
        )
    trainable = {url for url, ready in results.items() if ready}
    return len(urls), len(trainable), trainable


def _poll_forever() -> None:
    schedule = os.getenv("POLL_INTERVAL", "@hourly").strip()
    while not _stop_polling.wait(next_poll_delay(schedule)):
        try:
            _refresh_clients()
        except Exception:
            logger.exception("Could not refresh training services from the case slice")


def _poll_enabled() -> bool:
    value = os.getenv("POLL_ENABLED", "true").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("POLL_ENABLED must be true or false")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Weight aggregation service starting")
    poll_thread = None
    try:
        _refresh_clients()
    except Exception:
        logger.exception("Initial training service discovery failed")
    try:
        if _poll_enabled():
            # Validate the expression before starting the background thread.
            next_poll_delay(os.getenv("POLL_INTERVAL", "@hourly").strip())
            _stop_polling.clear()
            poll_thread = threading.Thread(
                target=_poll_forever,
                name="training-service-poller",
                daemon=True,
            )
            poll_thread.start()
    except Exception:
        logger.exception("Polling configuration is invalid")
    try:
        yield
    finally:
        logger.info("Weight aggregation service stopping")
        _stop_polling.set()
        if poll_thread:
            poll_thread.join(timeout=5)


app = FastAPI(
    title="Federated weight aggregation",
    version="0.1.0",
    lifespan=lifespan,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_status(payload: dict[str, Any]) -> None:
    _write_json(_status_path(), payload)


def _read_status() -> dict[str, Any]:
    if not _status_path().is_file():
        return {"status": "idle"}
    return json.loads(_status_path().read_text(encoding="utf-8"))


def _persist_session(session: ActiveSession, state: str) -> None:
    _write_json(
        _session_dir(session.session_id) / "session.json",
        {
            "session_id": session.session_id,
            "status": state,
            "expected_rounds": session.request.expected_rounds,
            "min_clients": session.request.min_clients,
            "round_timeout_seconds": session.request.round_timeout_seconds,
            "training_config": session.request.training_config.model_dump(mode="json"),
            "session_started_at": session.session_started_at,
            "round_started_at": session.round_started_at,
            "current_round": session.current_round,
            "all_clients": session.all_clients,
            "active_clients": session.active_clients,
            "failed_clients": session.failed_clients,
            "phase": session.phase,
            "aggregated_evaluation_metrics": session.aggregated_evaluation_metrics,
            "round_metrics": session.round_metrics,
            "model_signature": session.model_signature,
            "updated_at": _now(),
        },
    )


def _initial_global_weights() -> tuple[bytes, str, str]:
    path = _global_weights_path()
    if path.is_file():
        payload = path.read_bytes()
        signature = weight_signature(bytes_to_weights(payload))
        metadata_path = _global_metadata_path()
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("model_signature") != signature:
                raise RuntimeError(
                    "Persisted global weights do not match the current canonical model"
                )
        return payload, signature, "previous_session"
    weights, signature = create_initial_weights()
    return weights_to_bytes(weights), signature, "fresh_model"


def _client_post(base_url: str, path: str, message: BaseModel) -> dict:
    response = send_request(
        "POST",
        f"{base_url.rstrip('/')}{path}",
        json=message.model_dump(mode="json"),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _release_clients(session: ActiveSession) -> None:
    for client_id, base_url in session.all_clients.items():
        try:
            _client_post(
                base_url,
                CLIENT_SESSION_END_PATH,
                ClientSessionEnd(session_id=session.session_id, client_id=client_id),
            )
        except Exception:
            logger.exception("Could not release client %s at %s", client_id, base_url)


def _accept_client(
    session: ActiveSession,
    client_id: str,
    base_url: str,
) -> bool:
    try:
        response = _client_post(
            base_url,
            CLIENT_SESSION_START_PATH,
            ClientSessionStart(
                session_id=session.session_id,
                client_id=client_id,
                expected_rounds=session.request.expected_rounds,
                training_config=session.request.training_config,
            ),
        )
        return response["model_signature"] == session.model_signature
    except Exception:
        logger.exception("Client session start failed at %s", base_url)
        return False


def _reconcile_clients(session: ActiveSession) -> None:
    """Apply the latest registry snapshot at a boundary between rounds."""
    desired = _client_registry().snapshot()
    current_by_url = {url: client_id for client_id, url in session.active_clients.items()}

    for base_url in sorted(set(current_by_url) - desired):
        client_id = current_by_url[base_url]
        session.active_clients.pop(client_id, None)
        session.all_clients.pop(client_id, None)
        try:
            _client_post(
                base_url,
                CLIENT_SESSION_END_PATH,
                ClientSessionEnd(session_id=session.session_id, client_id=client_id),
            )
        except Exception:
            logger.exception("Could not release departed client at %s", base_url)

    for base_url in sorted(desired - set(current_by_url)):
        client_id = str(uuid.uuid4())
        if _accept_client(session, client_id, base_url):
            session.active_clients[client_id] = base_url
            session.all_clients[client_id] = base_url
        else:
            try:
                _client_post(
                    base_url,
                    CLIENT_SESSION_END_PATH,
                    ClientSessionEnd(session_id=session.session_id, client_id=client_id),
                )
            except Exception:
                logger.exception("Could not release rejected client at %s", base_url)


def _bootstrap_session(
    session_id: str,
    request: SessionStartRequest,
    candidate_urls: set[str],
    session_started_at: str,
) -> None:
    global _active
    assigned = {
        str(uuid.uuid4()): base_url
        for base_url in candidate_urls
    }
    logger.info(
        "Bootstrapping session %s with %d candidates (minimum=%d, rounds=%d)",
        session_id, len(candidate_urls), request.min_clients, request.expected_rounds,
    )
    initializing = ActiveSession(
        session_id=session_id,
        request=request,
        all_clients=assigned,
        active_clients={},
        model_signature="",
        session_started_at=session_started_at,
    )
    _persist_session(initializing, "initializing")
    accepted: dict[str, str] = {}
    signatures: dict[str, str] = {}
    for client_id, base_url in assigned.items():
        try:
            response = _client_post(
                base_url,
                CLIENT_SESSION_START_PATH,
                ClientSessionStart(
                    session_id=session_id,
                    client_id=client_id,
                    expected_rounds=request.expected_rounds,
                    training_config=request.training_config,
                ),
            )
            signatures[client_id] = response["model_signature"]
            accepted[client_id] = base_url
        except Exception as exc:
            logger.exception("Client session start failed at %s", base_url)

    try:
        global_weights, canonical_signature, source = _initial_global_weights()
        accepted = {
            client_id: base_url
            for client_id, base_url in accepted.items()
            if signatures[client_id] == canonical_signature
        }
        if len(accepted) < request.min_clients:
            raise RuntimeError(
                f"Only {len(accepted)} compatible clients started; "
                f"minimum is {request.min_clients}"
            )
        session = ActiveSession(
            session_id=session_id,
            request=request,
            all_clients=assigned,
            active_clients=accepted.copy(),
            model_signature=canonical_signature,
            global_weights=global_weights,
            session_started_at=session_started_at,
        )
        with _coordinator_lock:
            _active = session
            _in_progress_global_weights_path().parent.mkdir(parents=True, exist_ok=True)
            _in_progress_global_weights_path().write_bytes(global_weights)
            _write_status(
                {
                    "status": "running",
                    "session_id": session_id,
                    "current_round": 0,
                    "expected_rounds": request.expected_rounds,
                    "initial_weights_source": source,
                    "session_started_at": session.session_started_at,
                    "round_started_at": session.round_started_at,
                    "started_at": session.session_started_at,
                }
            )
            _persist_session(session, "running")
        logger.info("Session %s started with %d clients", session_id, len(accepted))
        _start_round(session)
    except Exception as exc:
        logger.exception("Could not bootstrap session %s", session_id)
        temporary = ActiveSession(
            session_id=session_id,
            request=request,
            all_clients=assigned,
            active_clients=accepted,
            model_signature="",
            session_started_at=session_started_at,
        )
        _release_clients(temporary)
        _in_progress_global_weights_path().unlink(missing_ok=True)
        with _coordinator_lock:
            _active = None
            _write_status(
                {
                    "status": "failed",
                    "session_id": session_id,
                    "error": str(exc),
                    "session_started_at": session_started_at,
                    "finished_at": _now(),
                }
            )


def _start_round(session: ActiveSession) -> None:
    with _coordinator_lock:
        if _active is not session:
            return
        _reconcile_clients(session)
        if len(session.active_clients) < session.request.min_clients:
            _finish_session(session, False, "Client count fell below min_clients")
            return
        session.current_round += 1
        session.round_started_at = _now()
        session.expected_clients = set(session.active_clients)
        session.weights_by_client = {}
        session.metrics_by_client = {}
        session.current_aggregated_weights = b""
        session.evaluation_metrics_by_client = {}
        session.aggregated_evaluation_metrics = {}
        session.phase = "training"
        session.round_metrics[str(session.current_round)] = {
            "round_id": session.current_round,
            "client_training": {},
            "server": {
                "peak_ram_bytes": _peak_ram_bytes(),
                "aggregation_seconds": 0.0,
            },
        }
        session.round_started_perf = time.perf_counter()
        round_dir = _session_dir(session.session_id) / f"round_{session.current_round}"
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "global-input.npz").write_bytes(session.global_weights)
        _persist_session(session, "running")
        _write_status(
            {
                "status": "running",
                "session_id": session.session_id,
                "current_round": session.current_round,
                "expected_rounds": session.request.expected_rounds,
                "active_clients": len(session.active_clients),
                "phase": session.phase,
                "session_started_at": session.session_started_at,
                "round_started_at": session.round_started_at,
                "started_at": session.round_started_at,
            }
        )
        logger.info(
            "Starting training round %d/%d for %d clients (session=%s)",
            session.current_round, session.request.expected_rounds,
            len(session.active_clients), session.session_id,
        )
    dispatch_started = time.perf_counter()
    failed = dispatch_round(
        session_id=session.session_id,
        round_id=session.current_round,
        client_urls={
            client_id: f"{base_url.rstrip('/')}{TRAIN_PATH}"
            for client_id, base_url in session.active_clients.items()
        },
        global_weights=session.global_weights,
        reply_url=_callback_url(),
    )
    round_metrics = session.round_metrics[str(session.current_round)]
    session.dispatch_finished_perf = time.perf_counter()
    round_metrics["server"]["dispatch_seconds"] = (
        session.dispatch_finished_perf - dispatch_started
    )
    round_metrics["server"]["collection_started_at"] = _now()
    for client_id in failed:
        _record_failure(session.session_id, session.current_round, client_id, "dispatch failed")
    with _coordinator_lock:
        if _active is session and session.expected_clients:
            timer = threading.Timer(
                session.request.round_timeout_seconds,
                _round_timeout,
                args=(session.session_id, session.current_round),
            )
            timer.daemon = True
            session.timer = timer
            timer.start()


def _record_failure(session_id: str, round_id: int, client_id: str, error: str) -> None:
    with _coordinator_lock:
        session = _active
        if not session or session.session_id != session_id or session.current_round != round_id:
            return
        if session.phase != "training":
            return
        if client_id not in session.expected_clients:
            return
        if client_id in session.weights_by_client:
            return
        session.expected_clients.discard(client_id)
        session.active_clients.pop(client_id, None)
        session.failed_clients[client_id] = error
        logger.warning(
            "Training failed for client %s in round %d (session=%s): %s",
            client_id, round_id, session_id, error,
        )
        _advance_if_ready(session)


def _record_result(result: TrainingResultMessage, weights: bytes) -> None:
    with _coordinator_lock:
        session = _active
        if not session or result.session_id != session.session_id:
            raise ValueError("Result belongs to no active session")
        if result.round_id != session.current_round:
            raise ValueError("Result belongs to a stale or future round")
        if session.phase != "training":
            raise ValueError("Session is not collecting training results")
        if result.client_id not in session.expected_clients:
            raise ValueError("Result is from an unexpected or removed client")
        if result.client_id in session.weights_by_client:
            return
        if weight_signature(bytes_to_weights(weights)) != session.model_signature:
            _record_failure(
                result.session_id,
                result.round_id,
                result.client_id,
                "model weight structure does not match the session",
            )
            return
        session.weights_by_client[result.client_id] = weights
        session.metrics_by_client[result.client_id] = result.metrics
        round_metrics = session.round_metrics.setdefault(
            str(session.current_round),
            {
                "round_id": session.current_round,
                "client_training": {},
                "server": {
                    "peak_ram_bytes": _peak_ram_bytes(),
                    "aggregation_seconds": 0.0,
                },
            },
        )
        if not session.round_started_perf:
            session.round_started_perf = time.perf_counter()
        if not session.dispatch_finished_perf:
            session.dispatch_finished_perf = session.round_started_perf
        round_metrics["client_training"][result.client_id] = result.metrics.model_dump(mode="json")
        logger.info(
            "Received training result from client %s for round %d (%d/%d)",
            result.client_id, result.round_id, len(session.weights_by_client),
            len(session.expected_clients),
        )
        round_dir = _session_dir(session.session_id) / f"round_{session.current_round}"
        client_dir = round_dir / "clients" / result.client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        (client_dir / "weights.npz").write_bytes(weights)
        _write_json(client_dir / "metrics.json", result.metrics.model_dump(mode="json"))
        aggregation_started = time.perf_counter()
        try:
            session.current_aggregated_weights = federated_average(
                session.weights_by_client,
                session.metrics_by_client,
            )
        except Exception as exc:
            _finish_session(session, False, f"Weight aggregation failed: {exc}")
            return
        round_metrics["server"]["aggregation_seconds"] += (
            time.perf_counter() - aggregation_started
        )
        round_metrics["server"]["peak_ram_bytes"] = _peak_ram_bytes()
        (round_dir / "aggregate-current.npz").write_bytes(
            session.current_aggregated_weights
        )
        live_path = _in_progress_global_weights_path()
        temporary = live_path.with_suffix(live_path.suffix + ".tmp")
        temporary.write_bytes(session.current_aggregated_weights)
        temporary.replace(live_path)
        _advance_if_ready(session)


def _advance_if_ready(session: ActiveSession) -> None:
    if len(session.active_clients) < session.request.min_clients:
        _finish_session(session, False, "Client count fell below min_clients")
        return
    if not session.expected_clients.issubset(session.weights_by_client):
        return
    if len(session.weights_by_client) < session.request.min_clients:
        _finish_session(session, False, "Too few successful client results")
        return
    if session.timer:
        session.timer.cancel()
        session.timer = None
    if not session.current_aggregated_weights:
        _finish_session(session, False, "Weight aggregation produced no result")
        return
    session.global_weights = session.current_aggregated_weights
    round_dir = _session_dir(session.session_id) / f"round_{session.current_round}"
    (round_dir / "aggregated.npz").write_bytes(session.global_weights)
    server_metrics = session.round_metrics[str(session.current_round)]["server"]
    client_totals = [
        metrics.get("total_seconds")
        for metrics in session.round_metrics[str(session.current_round)]["client_training"].values()
        if metrics.get("total_seconds") is not None
    ]
    training_round_seconds = time.perf_counter() - session.round_started_perf
    aggregation_seconds = server_metrics["aggregation_seconds"]
    server_metrics.update(
        {
            "training_round_seconds": training_round_seconds,
            "result_collection_seconds": time.perf_counter() - session.dispatch_finished_perf,
            "messaging_overhead_seconds": max(
                0.0,
                training_round_seconds - max(client_totals, default=0.0) - aggregation_seconds,
            ),
            "peak_ram_bytes": _peak_ram_bytes(),
        }
    )
    _start_evaluation(session)


def _start_evaluation(session: ActiveSession) -> None:
    with _coordinator_lock:
        if _active is not session:
            return
        session.phase = "evaluation"
        session.expected_clients = set(session.active_clients)
        session.evaluation_metrics_by_client = {}
        _persist_session(session, "running")
        _write_status(
            {
                "status": "running",
                "session_id": session.session_id,
                "current_round": session.current_round,
                "expected_rounds": session.request.expected_rounds,
                "active_clients": len(session.active_clients),
                "phase": session.phase,
                "session_started_at": session.session_started_at,
                "round_started_at": session.round_started_at,
                "started_at": session.round_started_at,
            }
        )
        logger.info(
            "Starting evaluation for round %d with %d clients (session=%s)",
            session.current_round, len(session.active_clients), session.session_id,
        )
    failed = dispatch_evaluation(
        session_id=session.session_id,
        round_id=session.current_round,
        client_urls={
            client_id: f"{base_url.rstrip('/')}{EVALUATE_PATH}"
            for client_id, base_url in session.active_clients.items()
        },
        global_weights=session.global_weights,
        reply_url=_evaluation_callback_url(),
    )
    for client_id in failed:
        _record_evaluation_failure(
            session.session_id,
            session.current_round,
            client_id,
            "evaluation dispatch failed",
        )
    with _coordinator_lock:
        if _active is session and session.expected_clients:
            timer = threading.Timer(
                session.request.round_timeout_seconds,
                _evaluation_timeout,
                args=(session.session_id, session.current_round),
            )
            timer.daemon = True
            session.timer = timer
            timer.start()


def _record_evaluation_failure(
    session_id: str,
    round_id: int,
    client_id: str,
    error: str,
) -> None:
    with _coordinator_lock:
        session = _active
        if (
            not session
            or session.session_id != session_id
            or session.current_round != round_id
            or session.phase != "evaluation"
        ):
            return
        if client_id not in session.expected_clients:
            return
        if client_id in session.evaluation_metrics_by_client:
            return
        session.expected_clients.discard(client_id)
        session.active_clients.pop(client_id, None)
        session.failed_clients[client_id] = error
        logger.warning(
            "Evaluation failed for client %s in round %d (session=%s): %s",
            client_id, round_id, session_id, error,
        )
        _advance_evaluation_if_ready(session)


def _record_evaluation_result(result: EvaluationResultMessage) -> None:
    with _coordinator_lock:
        session = _active
        if not session or result.session_id != session.session_id:
            raise ValueError("Evaluation belongs to no active session")
        if result.round_id != session.current_round:
            raise ValueError("Evaluation belongs to a stale or future round")
        if session.phase != "evaluation":
            raise ValueError("Session is not collecting evaluation results")
        if result.client_id not in session.expected_clients:
            raise ValueError("Evaluation is from an unexpected or removed client")
        if result.client_id in session.evaluation_metrics_by_client:
            return
        session.evaluation_metrics_by_client[result.client_id] = result.metrics
        logger.info(
            "Received evaluation from client %s for round %d (%d/%d)",
            result.client_id, result.round_id,
            len(session.evaluation_metrics_by_client), len(session.expected_clients),
        )
        try:
            session.aggregated_evaluation_metrics = aggregate_evaluation_metrics(
                session.evaluation_metrics_by_client
            )
        except Exception as exc:
            _finish_session(session, False, f"Evaluation aggregation failed: {exc}")
            return
        round_dir = _session_dir(session.session_id) / f"round_{session.current_round}"
        client_dir = round_dir / "clients" / result.client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            client_dir / "evaluation.json",
            result.metrics.model_dump(mode="json"),
        )
        _advance_evaluation_if_ready(session)


def _advance_evaluation_if_ready(session: ActiveSession) -> None:
    if len(session.active_clients) < session.request.min_clients:
        _finish_session(session, False, "Client count fell below min_clients during evaluation")
        return
    if not session.expected_clients.issubset(session.evaluation_metrics_by_client):
        return
    if len(session.evaluation_metrics_by_client) < session.request.min_clients:
        _finish_session(session, False, "Too few successful evaluation results")
        return
    if session.timer:
        session.timer.cancel()
        session.timer = None
    round_dir = _session_dir(session.session_id) / f"round_{session.current_round}"
    _write_json(
        round_dir / "aggregated-evaluation.json",
        session.aggregated_evaluation_metrics,
    )
    session.round_metrics[str(session.current_round)]["evaluation_metrics"] = (
        session.aggregated_evaluation_metrics
    )
    _write_json(
        _round_metrics_path(),
        {
            "session_id": session.session_id,
            "rounds": list(session.round_metrics.values()),
            "updated_at": _now(),
        },
    )
    _write_json(
        _data_dir() / "evaluation-metrics.json",
        {
            "session_id": session.session_id,
            "round_id": session.current_round,
            "metrics": session.aggregated_evaluation_metrics,
            "updated_at": _now(),
        },
    )
    if session.current_round >= session.request.expected_rounds:
        _finish_session(session, True)
    else:
        _start_round(session)


def _round_timeout(session_id: str, round_id: int) -> None:
    with _coordinator_lock:
        session = _active
        if not session or session.session_id != session_id or session.current_round != round_id:
            return
        if session.phase != "training":
            return
        missing = session.expected_clients - set(session.weights_by_client)
        for client_id in missing:
            session.active_clients.pop(client_id, None)
            session.failed_clients[client_id] = "round timeout"
        session.expected_clients -= missing
        _advance_if_ready(session)


def _evaluation_timeout(session_id: str, round_id: int) -> None:
    with _coordinator_lock:
        session = _active
        if (
            not session
            or session.session_id != session_id
            or session.current_round != round_id
            or session.phase != "evaluation"
        ):
            return
        missing = session.expected_clients - set(session.evaluation_metrics_by_client)
        for client_id in missing:
            session.active_clients.pop(client_id, None)
            session.failed_clients[client_id] = "evaluation timeout"
        session.expected_clients -= missing
        _advance_evaluation_if_ready(session)


def _finish_session(
    session: ActiveSession,
    success: bool,
    error: str | None = None,
) -> None:
    global _active
    if session.timer:
        session.timer.cancel()
        session.timer = None
    if success:
        path = _global_weights_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.tmp")
        temporary.write_bytes(session.global_weights)
        temporary.replace(path)
        _write_json(
            _global_metadata_path(),
            {
                "session_id": session.session_id,
                "model_signature": session.model_signature,
                "completed_rounds": session.current_round,
                "updated_at": _now(),
            },
        )
        _in_progress_global_weights_path().unlink(missing_ok=True)
    else:
        _in_progress_global_weights_path().unlink(missing_ok=True)
    state = "succeeded" if success else "failed"
    log = logger.info if success else logger.error
    log(
        "Session %s %s after %d rounds (failed_clients=%d, error=%s)",
        session.session_id, state, session.current_round,
        len(session.failed_clients), error,
    )
    _persist_session(session, state)
    _write_status(
        {
            "status": state,
            "session_id": session.session_id,
            "completed_rounds": session.current_round,
            "failed_clients": session.failed_clients,
            "evaluation_metrics": session.aggregated_evaluation_metrics,
            "session_started_at": session.session_started_at,
            "round_started_at": session.round_started_at,
            "error": error,
            "finished_at": _now(),
        }
    )
    _active = None
    threading.Thread(target=_release_clients, args=(session,), daemon=True).start()


def _recover_interrupted_session() -> None:
    current = _read_status()
    if current.get("status") not in {"initializing", "running"}:
        return
    session_id = current.get("session_id")
    session_path = _session_dir(session_id) / "session.json" if session_id else None
    if session_path and session_path.is_file():
        payload = json.loads(session_path.read_text(encoding="utf-8"))
        for client_id, base_url in payload.get("all_clients", {}).items():
            try:
                _client_post(
                    base_url,
                    CLIENT_SESSION_END_PATH,
                    ClientSessionEnd(session_id=session_id, client_id=client_id),
                )
            except Exception:
                logger.exception(
                    "Could not release client %s from interrupted session", client_id
                )
    _in_progress_global_weights_path().unlink(missing_ok=True)
    _write_status(
        {
            "status": "failed",
            "session_id": session_id,
            "error": "Aggregation service restarted during an active session",
            "session_started_at": current.get("session_started_at"),
            "round_started_at": current.get("round_started_at"),
            "finished_at": _now(),
        }
    )


@app.post("/session/start", status_code=status.HTTP_202_ACCEPTED)
def session_start(
    request: SessionStartRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    global _active
    request = request.model_copy(
        update={
            "training_config": request.training_config.model_copy(
                update={"num_rounds": request.expected_rounds}
            )
        }
    )
    with _coordinator_lock:
        if _active is None:
            _recover_interrupted_session()
        if _active is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A federated session is already active",
            )
        _, trainable_clients, candidate_urls = _client_counts()
        if request.min_clients > trainable_clients:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="min_clients exceeds trainable clients",
            )
        session_id = str(uuid.uuid4())
        session_started_at = _now()
        # Reserve the coordinator immediately while client bootstrap runs.
        _active = ActiveSession(
            session_id=session_id,
            request=request,
            all_clients={},
            active_clients={},
            model_signature="",
            session_started_at=session_started_at,
        )
        _write_status(
            {
                "status": "initializing",
                "session_id": session_id,
                "session_started_at": session_started_at,
                "round_started_at": None,
                "started_at": session_started_at,
            }
        )
    background_tasks.add_task(
        _bootstrap_session,
        session_id,
        request,
        candidate_urls,
        session_started_at,
    )
    return {"status": "initializing", "session_id": session_id}


@app.post("/training-results", status_code=status.HTTP_202_ACCEPTED)
async def training_results(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/"):
        form = await request.form()
        result = TrainingResultMessage.model_validate_json(str(form["message"]))
        upload = form["weights"]
        weights = await upload.read()
        background_tasks.add_task(_record_result, result, weights)
        return {"status": "accepted"}
    payload = await request.json()
    failure = TrainingFailureMessage.model_validate(payload)
    background_tasks.add_task(
        _record_failure,
        failure.session_id,
        failure.round_id,
        failure.client_id,
        failure.error,
    )
    return {"status": "accepted"}


@app.post("/evaluation-results", status_code=status.HTTP_202_ACCEPTED)
async def evaluation_results(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    payload = await request.json()
    if payload.get("type") == MessageType.EVALUATION_FAILURE.value:
        failure = EvaluationFailureMessage.model_validate(payload)
        background_tasks.add_task(
            _record_evaluation_failure,
            failure.session_id,
            failure.round_id,
            failure.client_id,
            failure.error,
        )
    else:
        result = EvaluationResultMessage.model_validate(payload)
        background_tasks.add_task(_record_evaluation_result, result)
    return {"status": "accepted"}


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readiness() -> dict[str, str]:
    try:
        directory = _data_dir()
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write-probe"
        probe.touch()
        probe.unlink()
        _client_registry()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready"}


@app.get("/status")
def session_status(include_round_metrics: bool = False) -> dict[str, Any]:
    result = _read_status()
    registered, trainable, _ = _client_counts()
    result.update(
        {
            "registered_clients": registered,
            "trainable_clients": trainable,
        }
    )
    if include_round_metrics:
        result["round_metrics"] = (
            json.loads(_round_metrics_path().read_text(encoding="utf-8"))
            if _round_metrics_path().is_file()
            else None
        )
    return result


@app.post("/refresh-clients")
def refresh_clients() -> dict[str, list[str]]:
    """Immediately refresh the client registry from the configured case slice."""
    return _refresh_clients()


@app.get("/weights", response_class=FileResponse)
def global_weights() -> FileResponse:
    """Download the latest global model, including an active session checkpoint."""
    with _coordinator_lock:
        live_path = _in_progress_global_weights_path()
        path = live_path if live_path.is_file() else _global_weights_path()
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed global weights are available",
        )
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename="global-weights.npz",
    )


@app.get("/evaluation-metrics")
def aggregated_evaluation_metrics() -> dict[str, Any]:
    path = _data_dir() / "evaluation-metrics.json"
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No aggregated evaluation metrics are available",
        )
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
