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
from .train import run_evaluation, run_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def _round_dir(session_id: str, client_id: str, round_id: int) -> Path:
    return (
        _data_dir()
        / "model-training"
        / "rounds"
        / session_id
        / client_id
        / f"round_{round_id}"
    )


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
    directory = _round_dir(session_id, client_id, round_id)
    message_path = directory / "result.json"
    weights_path = directory / "weights.npz"
    if not message_path.is_file() or not weights_path.is_file():
        return None
    return (
        TrainingResultMessage.model_validate_json(message_path.read_text(encoding="utf-8")),
        weights_path.read_bytes(),
    )


def _cache_result(result: TrainingResultMessage, weights: bytes) -> None:
    directory = _round_dir(result.session_id, result.client_id, result.round_id)
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
    path = _round_dir(session_id, client_id, round_id) / "evaluation.json"
    if not path.is_file():
        return None
    return EvaluationResultMessage.model_validate_json(path.read_text(encoding="utf-8"))


def _cache_evaluation(result: EvaluationResultMessage) -> None:
    directory = _round_dir(result.session_id, result.client_id, result.round_id)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "evaluation.json.tmp"
    temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(directory / "evaluation.json")


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
    _write_status(
        {
            "status": "session_active",
            "session_id": request.session_id,
            "client_id": request.client_id,
            "expected_rounds": request.expected_rounds,
            "started_at": _now(),
        }
    )
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
    _session_path().unlink(missing_ok=True)
    _write_status(
        {
            "status": "session_ended",
            "session_id": request.session_id,
            "client_id": request.client_id,
            "finished_at": _now(),
        }
    )
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
                "started_at": _now(),
                "finished_at": None,
            }
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
                "started_at": _now(),
                "finished_at": None,
            }
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
    try:
        result, local_weights = run_training(msg, config, global_weights)
    except Exception as exc:
        logger.exception("Training failed for round %d", msg.round_id)
        _write_status(
            {
                "status": "failed",
                "round_id": msg.round_id,
                "client_id": msg.client_id,
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
        _cache_result(result, local_weights)
        _write_status(
            {
                "status": "reporting",
                "round_id": msg.round_id,
                "client_id": msg.client_id,
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
                    "round_id": msg.round_id,
                    "client_id": msg.client_id,
                    "finished_at": _now(),
                    "error": str(exc),
                }
            )
        else:
            logger.info("Reported result for round %d to %s", msg.round_id, msg.reply_url)
            _write_status(
                {
                    "status": "succeeded",
                    "round_id": msg.round_id,
                    "client_id": msg.client_id,
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
    try:
        result = run_evaluation(msg, config, global_weights)
    except Exception as exc:
        logger.exception("Evaluation failed for round %d", msg.round_id)
        _write_status(
            {
                "status": "evaluation_failed",
                "round_id": msg.round_id,
                "client_id": msg.client_id,
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
        _cache_evaluation(result)
        try:
            post_message(msg.reply_url, result)
        except Exception as exc:
            logger.exception("Could not report evaluation for round %d", msg.round_id)
            _write_status(
                {
                    "status": "evaluation_report_failed",
                    "round_id": msg.round_id,
                    "client_id": msg.client_id,
                    "finished_at": _now(),
                    "error": str(exc),
                }
            )
        else:
            _write_status(
                {
                    "status": "evaluation_succeeded",
                    "round_id": msg.round_id,
                    "client_id": msg.client_id,
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
    _, weights_path = _latest_result_paths()
    return FileResponse(
        weights_path,
        media_type="application/octet-stream",
        filename="weights.npz",
    )


@app.get("/metrics")
def training_metrics() -> dict[str, Any]:
    result_path, _ = _latest_result_paths()
    result = TrainingResultMessage.model_validate_json(
        result_path.read_text(encoding="utf-8")
    )
    return {
        "session_id": result.session_id,
        "round_id": result.round_id,
        "client_id": result.client_id,
        "metrics": result.metrics.model_dump(),
    }


@app.get("/evaluation-metrics")
def evaluation_metrics() -> dict[str, Any]:
    rounds_dir = _data_dir() / "model-training" / "rounds"
    candidates = list(rounds_dir.glob("*/*/round_*/evaluation.json"))
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed evaluation output is available",
        )
    path = max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)
    result = EvaluationResultMessage.model_validate_json(path.read_text(encoding="utf-8"))
    return result.model_dump(mode="json")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
