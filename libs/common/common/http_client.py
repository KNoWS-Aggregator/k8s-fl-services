"""REST client for sending control messages, optionally with a binary
weights payload attached as a multipart field (see messages.py)."""
from __future__ import annotations

import base64
import logging
import os
import time

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

WEIGHTS_FIELD = "weights"
MESSAGE_FIELD = "message"


def _egress_uma_url() -> str:
    return os.getenv("EGRESS_UMA_URL", "").rstrip("/")


def request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json: object | None = None,
    data: dict[str, str] | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
    content: bytes | str | None = None,
    timeout: float = 60.0,
) -> httpx.Response:
    """Send a request directly or tunnel it through the UMA egress proxy."""
    with httpx.Client(timeout=timeout) as client:
        outbound = client.build_request(
            method,
            url,
            headers=headers,
            json=json,
            data=data,
            files=files,
            content=content,
        )
        egress_uma_url = _egress_uma_url()
        if not egress_uma_url:
            return client.send(outbound)

        payload = {
            "url": str(outbound.url),
            "method": outbound.method,
            "headers": {
                key: value for key, value in outbound.headers.items() if key.lower() != "host"
            },
        }
        body = outbound.read()
        if body:
            payload["bodyBase64"] = base64.b64encode(body).decode("ascii")

        proxied = client.build_request(
            "POST",
            f"{egress_uma_url}/fetch",
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        return client.send(proxied)


def post_message(url: str, message: BaseModel, weights: bytes | None = None,
                  timeout: float = 60.0, retries: int = 3, backoff: float = 2.0) -> httpx.Response:
    """POST a JSON control message, optionally with a binary weights part."""
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            if weights is None:
                response = request(
                    "POST",
                    url,
                    json=message.model_dump(mode="json"),
                    timeout=timeout,
                )
            else:
                files = {WEIGHTS_FIELD: ("weights.npz", weights, "application/octet-stream")}
                data = {MESSAGE_FIELD: message.model_dump_json()}
                response = request(
                    "POST",
                    url,
                    data=data,
                    files=files,
                    timeout=timeout,
                )
            response.raise_for_status()
            return response
        except (httpx.HTTPError, httpx.TransportError) as exc:
            last_exc = exc
            logger.warning("POST %s failed (attempt %d/%d): %s", url, attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)

    raise RuntimeError(f"Failed to POST message to {url} after {retries} attempts") from last_exc
