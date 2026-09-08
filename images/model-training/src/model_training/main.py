"""Entrypoint for the Model Training Service.

A single long-running Deployment that listens for TrainingInitMessage
requests and runs training itself - no separate Job container. Training
runs as a background task so the HTTP response isn't blocked for the
duration of a round; the actual result is POSTed to reply_url once done.
"""
import logging
import os
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from common.http_client import post_message
from common.messages import (
    ClientSessionEnd,
    ClientSessionStart,
    EvaluationFailureMessage,
    EvaluationInitMessage,
    EvaluationResultMessage,
    TrainingFailureMessage,
    TrainingInitMessage,
    TrainingResultMessage,
)
from fl_model import create_initial_weights
from .support import round_dir
from .train import run_evaluation, run_training

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

app = FastAPI(title="Federated model training", version="0.1.0")
_run_lock = threading.Lock()
_state_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "/app/data"))


def _status_path() -> Path:
    return _data_dir() / "model-training" / "status.json"


def _session_path() -> Path:
    return _data_dir() / "model-training" / "session.json"


def _read_status() -> dict[str, Any]:
    path = _status_path()
    if not path.is_file():
        return {"status": "idle"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read persistent training status")
        return {"status": "unknown", "error": "Persistent status is unreadable"}


def _write_status(payload: dict[str, Any]) -> None:
    with _state_lock:
        path = _status_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)


def _status_timestamps(
    session_id: str,
    round_id: int | None = None,
    *,
    new_round: bool = False,
) -> dict[str, str | None]:
    current = _read_status()
    same_session = current.get("session_id") == session_id
    now = _now()
    session_started_at = (
        current.get("session_started_at") if same_session else None
    ) or now
    same_round = same_session and current.get("round_id") == round_id
    round_started_at = (
        now
        if new_round
        else current.get("round_started_at") if same_round or round_id is None else None
    )
    if round_id is not None and round_started_at is None:
        round_started_at = now
    return {
        "session_started_at": session_started_at,
        "round_started_at": round_started_at,
    }


def _prepared_data_status() -> dict[str, Any]:
    data_dir = _data_dir()
    paths = (
        data_dir / "downloads" / "accel.parquet",
        data_dir / "downloads" / "gt.parquet",
    )
    files_available = all(path.is_file() for path in paths)
    state_path = data_dir / "dataset-state.json"
    manifest_path = data_dir / "manifest.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    state_status = state.get("status", "legacy" if files_available else "missing")
    generation = (
        state.get("generation")
        if state_status == "ready"
        else manifest.get("generation")
    )
    valid = files_available and bool(generation or not state_path.is_file())
    return {
        "valid": valid,
        "status": state_status,
        "generation": generation,
    }


def _read_session() -> ClientSessionStart | None:
    path = _session_path()
    if not path.is_file():
        return None
    return ClientSessionStart.model_validate_json(path.read_text(encoding="utf-8"))


def _write_session(session: ClientSessionStart) -> None:
    path = _session_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(session.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def _cached_result(
    session_id: str,
    client_id: str,
    round_id: int,
) -> tuple[TrainingResultMessage, bytes] | None:
    directory = round_dir(session_id, client_id, round_id)
    message_path = directory / "result.json"
    weights_path = directory / "weights.npz"
    if not message_path.is_file() or not weights_path.is_file():
        return None
    return (
        TrainingResultMessage.model_validate_json(message_path.read_text(encoding="utf-8")),
        weights_path.read_bytes(),
    )


def _cache_result(result: TrainingResultMessage, weights: bytes) -> None:
    directory = round_dir(result.session_id, result.client_id, result.round_id)
    directory.mkdir(parents=True, exist_ok=True)
    weights_temporary = directory / "weights.npz.tmp"
    result_temporary = directory / "result.json.tmp"
    weights_temporary.write_bytes(weights)
    result_temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    weights_temporary.replace(directory / "weights.npz")
    result_temporary.replace(directory / "result.json")


def _cached_evaluation(
    session_id: str,
    client_id: str,
    round_id: int,
) -> EvaluationResultMessage | None:
    path = round_dir(session_id, client_id, round_id) / "evaluation.json"
    if not path.is_file():
        return None
    return EvaluationResultMessage.model_validate_json(path.read_text(encoding="utf-8"))


def _cache_evaluation(result: EvaluationResultMessage) -> None:
    directory = round_dir(result.session_id, result.client_id, result.round_id)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "evaluation.json.tmp"
    temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(directory / "evaluation.json")


def _latest_round_dir() -> Path:
    current = _read_status()
    if current.get("status") == "running":
        directory = round_dir(
            current["session_id"], current["client_id"], current["round_id"]
        )
        if (directory / "weights.npz").is_file():
            return directory
    rounds_dir = _data_dir() / "model-training" / "rounds"
    candidates = [
        path.parent
        for path in rounds_dir.glob("*/*/round_*/weights.npz")
        if path.is_file()
    ]
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No training output is available",
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _latest_result_paths() -> tuple[Path, Path]:
    rounds_dir = _data_dir() / "model-training" / "rounds"
    candidates = list(rounds_dir.glob("*/*/round_*/result.json"))
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed training output is available",
        )
    result_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    weights_path = result_path.with_name("weights.npz")
    if not weights_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The latest training result has no corresponding weights",
        )
    return result_path, weights_path


def _latest_metrics_dir() -> Path:
    rounds_dir = _data_dir() / "model-training" / "rounds"
    candidates = {
        path.parent
        for pattern in ("result.json", "evaluation.json")
        for path in rounds_dir.glob(f"*/*/round_*/{pattern}")
        if path.is_file()
    }
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed training or evaluation metrics are available",
        )
    return max(
        candidates,
        key=lambda directory: max(
            path.stat().st_mtime_ns
            for path in (directory / "result.json", directory / "evaluation.json")
            if path.is_file()
        ),
    )


def _round_history_dirs() -> list[Path]:
    rounds_dir = _data_dir() / "model-training" / "rounds"
    paths = []
    for path in rounds_dir.glob("*/*/round_*"):
        try:
            int(path.name.removeprefix("round_"))
        except ValueError:
            continue
        paths.append(path)
    current = _read_status()
    if current.get("status") == "running":
        active_dir = round_dir(
            current["session_id"], current["client_id"], current["round_id"]
        )
        if active_dir not in paths:
            paths.append(active_dir)
    return sorted(
        paths,
        key=lambda path: (
            path.parents[1].name,
            path.parent.name,
            int(path.name.removeprefix("round_")),
        ),
    )


def _session_for(msg: TrainingInitMessage) -> ClientSessionStart:
    session = _read_session()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training client has no active session",
        )
    if msg.client_id != session.client_id or msg.session_id != session.session_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training message does not match the active session assignment",
        )
    if msg.round_id > session.expected_rounds:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Round exceeds the active session's expected rounds",
        )
    return session


def _evaluation_session_for(msg: EvaluationInitMessage) -> ClientSessionStart:
    session = _read_session()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training client has no active session",
        )
    if msg.client_id != session.client_id or msg.session_id != session.session_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Evaluation message does not match the active session assignment",
        )
    if msg.round_id > session.expected_rounds:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Round exceeds the active session's expected rounds",
        )
    return session


@app.post("/session/start")
def session_start(request: ClientSessionStart) -> dict[str, Any]:
    logger.info(
        "Session start requested (session=%s, client=%s, rounds=%d)",
        request.session_id, request.client_id, request.expected_rounds,
    )
    prepared_data = _prepared_data_status()
    if not prepared_data["valid"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No valid prepared dataset is available",
        )
    if _run_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A training round is currently running",
        )
    current = _read_session()
    if current:
        if current == request:
            _, signature = create_initial_weights()
            return {
                "status": "already_started",
                "session_id": request.session_id,
                "client_id": request.client_id,
                "model_signature": signature,
            }
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training client is assigned to another active session",
        )
    _write_session(request)
    _, signature = create_initial_weights()
    session_started_at = _now()
    _write_status(
        {
            "status": "session_active",
            "session_id": request.session_id,
            "client_id": request.client_id,
            "expected_rounds": request.expected_rounds,
            "session_started_at": session_started_at,
            "round_started_at": None,
        }
    )
    logger.info("Session started (session=%s, client=%s)", request.session_id, request.client_id)
    return {
        "status": "started",
        "session_id": request.session_id,
        "client_id": request.client_id,
        "model_signature": signature,
    }


@app.post("/session/end")
def session_end(request: ClientSessionEnd) -> dict[str, str]:
    current = _read_session()
    if not current:
        return {"status": "idle"}
    if current.session_id != request.session_id or current.client_id != request.client_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session end request does not match the active assignment",
        )
    timestamps = _status_timestamps(request.session_id)
    _session_path().unlink(missing_ok=True)
    _write_status(
        {
            "status": "session_ended",
            "session_id": request.session_id,
            "client_id": request.client_id,
            **timestamps,
            "finished_at": _now(),
        }
    )
    logger.info("Session ended (session=%s, client=%s)", request.session_id, request.client_id)
    return {"status": "ended"}


@app.post("/train")
async def train_endpoint(
    background_tasks: BackgroundTasks,
    message: str = Form(...),
    weights: UploadFile = File(...),
):
    msg = TrainingInitMessage.model_validate_json(message)
    session = _session_for(msg)

    cached = _cached_result(msg.session_id, msg.client_id, msg.round_id)
    if cached:
        logger.info("Round %d already completed; reporting cached result", msg.round_id)
        background_tasks.add_task(_report_cached, msg.reply_url, cached)
        return {
            "status": "already_completed",
            "round_id": msg.round_id,
            "client_id": msg.client_id,
        }

    if not _run_lock.acquire(blocking=False):
        current = _read_status()
        if (
            current.get("status") == "running"
            and current.get("round_id") == msg.round_id
            and current.get("client_id") == msg.client_id
        ):
            return {
                "status": "already_running",
                "round_id": msg.round_id,
                "client_id": msg.client_id,
            }
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another training round is already running",
        )

    try:
        global_weights = await weights.read()
        if not global_weights:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Weights upload is empty",
            )
        _write_status(
            {
                "status": "running",
                "session_id": msg.session_id,
                "round_id": msg.round_id,
                "client_id": msg.client_id,
                **_status_timestamps(msg.session_id, msg.round_id, new_round=True),
                "finished_at": None,
            }
        )
        logger.info(
            "Accepted training round %d (session=%s, client=%s, weights_bytes=%d)",
            msg.round_id, msg.session_id, msg.client_id, len(global_weights),
        )
        background_tasks.add_task(
            _run_and_report,
            msg,
            session.training_config,
            global_weights,
        )
    except Exception:
        _run_lock.release()
        raise
    return {"status": "accepted", "round_id": msg.round_id, "client_id": msg.client_id}


@app.post("/evaluate")
async def evaluate_endpoint(
    background_tasks: BackgroundTasks,
    message: str = Form(...),
    weights: UploadFile = File(...),
):
    msg = EvaluationInitMessage.model_validate_json(message)
    session = _evaluation_session_for(msg)
    cached = _cached_evaluation(msg.session_id, msg.client_id, msg.round_id)
    if cached:
        logger.info("Round %d evaluation already completed; reporting cached result", msg.round_id)
        background_tasks.add_task(_report_cached_evaluation, msg.reply_url, cached)
        return {
            "status": "already_completed",
            "round_id": msg.round_id,
            "client_id": msg.client_id,
        }
    if not _run_lock.acquire(blocking=False):
        current = _read_status()
        if (
            current.get("status") == "evaluating"
            and current.get("round_id") == msg.round_id
            and current.get("client_id") == msg.client_id
        ):
            return {
                "status": "already_running",
                "round_id": msg.round_id,
                "client_id": msg.client_id,
            }
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another training or evaluation operation is already running",
        )
    try:
        global_weights = await weights.read()
        if not global_weights:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Weights upload is empty",
            )
        _write_status(
            {
                "status": "evaluating",
                "session_id": msg.session_id,
                "round_id": msg.round_id,
                "client_id": msg.client_id,
                **_status_timestamps(msg.session_id, msg.round_id),
                "finished_at": None,
            }
        )
        logger.info(
            "Accepted evaluation round %d (session=%s, client=%s, weights_bytes=%d)",
            msg.round_id, msg.session_id, msg.client_id, len(global_weights),
        )
        background_tasks.add_task(
            _run_evaluation_and_report,
            msg,
            session.training_config,
            global_weights,
        )
    except Exception:
        _run_lock.release()
        raise
    return {"status": "accepted", "round_id": msg.round_id, "client_id": msg.client_id}


def _run_and_report(msg: TrainingInitMessage, config, global_weights: bytes) -> None:
    logger.info("Training round %d started for client %s", msg.round_id, msg.client_id)
    try:
        result, local_weights = run_training(msg, config, global_weights)
    except Exception as exc:
        logger.exception("Training failed for round %d", msg.round_id)
        _write_status(
            {
                "status": "failed",
                "session_id": msg.session_id,
                "round_id": msg.round_id,
                "client_id": msg.client_id,
                **_status_timestamps(msg.session_id, msg.round_id),
                "finished_at": _now(),
                "error": str(exc),
            }
        )
        failure = TrainingFailureMessage(
            session_id=msg.session_id,
            round_id=msg.round_id,
            client_id=msg.client_id,
            error=str(exc),
        )
        try:
            post_message(msg.reply_url, failure)
        except Exception:
            logger.exception("Could not report failure for round %d", msg.round_id)
    else:
        logger.info(
            "Training round %d completed (examples=%d, loss=%s, accuracy=%s, weighted_f1=%s, val_loss=%s, val_accuracy=%s, val_weighted_f1=%s)",
            msg.round_id, result.metrics.num_examples,
            result.metrics.train_loss, result.metrics.train_accuracy,
            result.metrics.train_f1_weighted, result.metrics.val_loss,
            result.metrics.val_accuracy, result.metrics.val_f1_weighted,
        )
        _cache_result(result, local_weights)
        _write_status(
            {
                "status": "reporting",
                "session_id": msg.session_id,
                "round_id": msg.round_id,
                "client_id": msg.client_id,
                **_status_timestamps(msg.session_id, msg.round_id),
                "finished_at": _now(),
            }
        )
        try:
            post_message(msg.reply_url, result, weights=local_weights)
        except Exception as exc:
            logger.exception("Could not report result for round %d", msg.round_id)
            _write_status(
                {
                    "status": "report_failed",
                    "session_id": msg.session_id,
                    "round_id": msg.round_id,
                    "client_id": msg.client_id,
                    **_status_timestamps(msg.session_id, msg.round_id),
                    "finished_at": _now(),
                    "error": str(exc),
                }
            )
        else:
            logger.info("Reported result for round %d to %s", msg.round_id, msg.reply_url)
            _write_status(
                {
                    "status": "succeeded",
                    "session_id": msg.session_id,
                    "round_id": msg.round_id,
                    "client_id": msg.client_id,
                    **_status_timestamps(msg.session_id, msg.round_id),
                    "finished_at": _now(),
                }
            )
    finally:
        _run_lock.release()


def _report_cached(
    reply_url: str,
    cached: tuple[TrainingResultMessage, bytes],
) -> None:
    result, weights = cached
    try:
        post_message(reply_url, result, weights=weights)
        logger.info("Re-reported cached result for round %d", result.round_id)
    except Exception:
        logger.exception("Could not re-report cached result for round %d", result.round_id)


def _run_evaluation_and_report(
    msg: EvaluationInitMessage,
    config,
    global_weights: bytes,
) -> None:
    logger.info("Evaluation round %d started for client %s", msg.round_id, msg.client_id)
    try:
        result = run_evaluation(msg, config, global_weights)
    except Exception as exc:
        logger.exception("Evaluation failed for round %d", msg.round_id)
        _write_status(
            {
                "status": "evaluation_failed",
                "session_id": msg.session_id,
                "round_id": msg.round_id,
                "client_id": msg.client_id,
                **_status_timestamps(msg.session_id, msg.round_id),
                "finished_at": _now(),
                "error": str(exc),
            }
        )
        failure = EvaluationFailureMessage(
            session_id=msg.session_id,
            round_id=msg.round_id,
            client_id=msg.client_id,
            error=str(exc),
        )
        try:
            post_message(msg.reply_url, failure)
        except Exception:
            logger.exception("Could not report evaluation failure for round %d", msg.round_id)
    else:
        logger.info(
            "Evaluation round %d completed (examples=%d, loss=%s, accuracy=%s, macro_f1=%s, weighted_f1=%s, macro_precision=%s, macro_recall=%s)",
            msg.round_id, result.metrics.num_examples,
            result.metrics.eval_loss, result.metrics.eval_accuracy,
            result.metrics.eval_f1_macro, result.metrics.eval_f1_weighted,
            result.metrics.eval_precision_macro, result.metrics.eval_recall_macro,
        )
        _cache_evaluation(result)
        try:
            post_message(msg.reply_url, result)
        except Exception as exc:
            logger.exception("Could not report evaluation for round %d", msg.round_id)
            _write_status(
                {
                    "status": "evaluation_report_failed",
                    "session_id": msg.session_id,
                    "round_id": msg.round_id,
                    "client_id": msg.client_id,
                    **_status_timestamps(msg.session_id, msg.round_id),
                    "finished_at": _now(),
                    "error": str(exc),
                }
            )
        else:
            _write_status(
                {
                    "status": "evaluation_succeeded",
                    "session_id": msg.session_id,
                    "round_id": msg.round_id,
                    "client_id": msg.client_id,
                    **_status_timestamps(msg.session_id, msg.round_id),
                    "finished_at": _now(),
                }
            )
    finally:
        _run_lock.release()


def _report_cached_evaluation(
    reply_url: str,
    result: EvaluationResultMessage,
) -> None:
    try:
        post_message(reply_url, result)
        logger.info("Re-reported cached evaluation for round %d", result.round_id)
    except Exception:
        logger.exception(
            "Could not re-report cached evaluation for round %d", result.round_id
        )


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readiness() -> dict[str, str]:
    try:
        data_dir = _data_dir()
        if not _prepared_data_status()["valid"]:
            raise RuntimeError("No valid prepared dataset is available")
        work_dir = data_dir / "model-training"
        work_dir.mkdir(parents=True, exist_ok=True)
        probe = work_dir / ".write-probe"
        probe.touch()
        probe.unlink()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return {"status": "ready"}


@app.get("/status")
def training_status() -> dict[str, Any]:
    result = _read_status()
    session = _read_session()
    prepared_data = _prepared_data_status()
    result.update(
        {
            "prepared_data": prepared_data,
            "session": {
                "active": session is not None,
                "session_id": session.session_id if session else None,
            },
            "trainable": prepared_data["valid"] and session is None,
        }
    )
    return result


@app.get("/weights", response_class=FileResponse)
def training_weights() -> FileResponse:
    """Return the newest available local weights, including a live checkpoint."""
    weights_path = _latest_round_dir() / "weights.npz"
    return FileResponse(
        weights_path,
        media_type="application/octet-stream",
        filename="weights.npz",
    )


@app.get("/metrics")
def training_metrics() -> dict[str, Any]:
    """Return local training and evaluation metrics for the latest round."""
    directory = _latest_metrics_dir()
    result_path = directory / "result.json"
    evaluation_path = directory / "evaluation.json"
    result = (
        TrainingResultMessage.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        if result_path.is_file()
        else None
    )
    evaluation = (
        EvaluationResultMessage.model_validate_json(
            evaluation_path.read_text(encoding="utf-8")
        )
        if evaluation_path.is_file()
        else None
    )
    context = result or evaluation
    assert context is not None
    return {
        "session_id": context.session_id,
        "round_id": context.round_id,
        "client_id": context.client_id,
        "training": result.metrics.model_dump(mode="json") if result else None,
        "evaluation": evaluation.metrics.model_dump(mode="json") if evaluation else None,
    }


@app.get("/metrics/history")
def training_history() -> dict[str, Any]:
    """Return per-epoch training history for every stored local round."""
    round_dirs = _round_history_dirs()
    if not round_dirs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No training history is available",
        )
    return {
        "rounds": [
            {
                "session_id": round_dir.parents[1].name,
                "client_id": round_dir.parent.name,
                "round_id": int(round_dir.name.removeprefix("round_")),
                "history": json.loads((round_dir / "history.json").read_text(encoding="utf-8"))
                if (round_dir / "history.json").is_file()
                else [],
            }
            for round_dir in round_dirs
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
