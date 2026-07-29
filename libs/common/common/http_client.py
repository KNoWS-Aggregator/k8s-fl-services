"""REST client for sending control messages, optionally with a binary
weights payload attached as a multipart field (see messages.py)."""
from __future__ import annotations

import logging
import time

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

WEIGHTS_FIELD = "weights"
MESSAGE_FIELD = "message"


def post_message(url: str, message: BaseModel, weights: bytes | None = None,
                  timeout: float = 60.0, retries: int = 3, backoff: float = 2.0) -> httpx.Response:
    """POST a JSON control message, optionally with a binary weights part."""
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            if weights is None:
                response = httpx.post(url, json=message.model_dump(mode="json"), timeout=timeout)
            else:
                files = {WEIGHTS_FIELD: ("weights.npz", weights, "application/octet-stream")}
                data = {MESSAGE_FIELD: message.model_dump_json()}
                response = httpx.post(url, data=data, files=files, timeout=timeout)
            response.raise_for_status()
            return response
        except (httpx.HTTPError, httpx.TransportError) as exc:
            last_exc = exc
            logger.warning("POST %s failed (attempt %d/%d): %s", url, attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)

    raise RuntimeError(f"Failed to POST message to {url} after {retries} attempts") from last_exc