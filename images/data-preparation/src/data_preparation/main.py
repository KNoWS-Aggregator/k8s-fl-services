"""HTTP entrypoint for data preparation."""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import uvicorn
from croniter import croniter
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse

from .converter import Converter
from .settings import Settings

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

_lock = threading.Lock()
_state: dict[str, Any] = {"status": "idle", "started_at": None, "finished_at": None}
_stop_polling = threading.Event()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    public = dict(result)
    changes = result.get("changes")
    if isinstance(changes, dict):
        public["changes"] = {
            name: len(participants) if isinstance(participants, list) else participants
            for name, participants in changes.items()
        }
    return public


def _prepare() -> None:
    global _state
    try:
        settings = Settings.from_env()
        logger.info("Starting data preparation (data_dir=%s)", settings.data_dir)
        result = Converter(settings).run()
        _state = {
            **_state,
            "status": "succeeded",
            "finished_at": _now(),
            "result": _public_result(result),
        }
        logger.info(
            "Data preparation succeeded in %.2fs (participants=%d, published=%s)",
            result["total_seconds"],
            result["participants_discovered"],
            result["published"],
        )
    except Exception as exc:
        logger.exception("Data preparation failed")
        _state = {**_state, "status": "failed", "finished_at": _now(), "error": str(exc)}
    finally:
        _lock.release()

def _start_preparation() -> bool:
    global _state
    if not _lock.acquire(blocking=False):
        logger.info("Data preparation trigger ignored because a run is already active")
        return False
    _state = {"status": "running", "started_at": _now(), "finished_at": None}
    threading.Thread(target=_prepare, name="data-preparation", daemon=True).start()
    return True


def _poll_forever(settings: Settings) -> None:
    global _state
    schedule = croniter(settings.poll_interval, datetime.now(timezone.utc))
    while not _stop_polling.is_set():
        next_poll = schedule.get_next(datetime)
        logger.info("Next data preparation poll scheduled for %s", next_poll.isoformat())
        _state = {**_state, "next_poll_at": next_poll.isoformat()}
        delay = max(0.0, (next_poll - datetime.now(timezone.utc)).total_seconds())
        if _stop_polling.wait(delay):
            break
        _start_preparation()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Data preparation service starting")
    _start_preparation()
    poll_thread = None
    try:
        settings = Settings.from_env()
        if settings.poll_enabled:
            logger.info("Periodic polling enabled (schedule=%s)", settings.poll_interval)
            _stop_polling.clear()
            poll_thread = threading.Thread(
                target=_poll_forever,
                args=(settings,),
                name="data-preparation-poller",
                daemon=True,
            )
            poll_thread.start()
    except ValueError:
        logger.exception("Polling configuration is invalid")
    try:
        yield
    finally:
        logger.info("Data preparation service stopping")
        _stop_polling.set()
        if poll_thread:
            poll_thread.join(timeout=5)


app = FastAPI(
    title="Kvasir data preparation",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readiness() -> dict[str, str]:
    try:
        settings = Settings.from_env()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.data_dir / ".write-probe"
        probe.touch()
        probe.unlink()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return {"status": "ready"}


@app.get("/status")
def preparation_status() -> dict[str, Any]:
    return dict(_state)

@app.get("/results", response_class=FileResponse)
def download_results() -> FileResponse:
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    archive = settings.result_archive_path
    if not archive.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No aggregated results are available yet",
        )
    return FileResponse(
        archive,
        media_type="application/zip",
        filename=settings.result_archive_name,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
