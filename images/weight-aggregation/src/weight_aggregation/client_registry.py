"""Polling registry for training services exposed by a researcher case slice."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from urllib.parse import urlparse

from common.http_client import request
from croniter import croniter


class ClientRegistry:
    def __init__(self, case_slice: str):
        self.case_slice = case_slice.rstrip("/")
        parsed = urlparse(self.case_slice)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("CASE_SLICE must be an absolute URL")
        self._lock = threading.Lock()
        self._urls: set[str] = set()

    def refresh(self) -> dict[str, list[str]]:
        response = request(
            "POST",
            f"{self.case_slice}/query",
            headers={"Content-Type": "application/json"},
            json={"query": "{ case { id trainingServices } }"},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            messages = "; ".join(
                error.get("message", str(error)) for error in payload["errors"]
            )
            raise RuntimeError(f"Case slice query failed: {messages}")
        values = ((payload.get("data") or {}).get("case") or {}).get(
            "trainingServices"
        ) or []
        discovered = {self._validate_url(value) for value in values}
        with self._lock:
            previous = self._urls
            self._urls = discovered
        return {
            "added": sorted(discovered - previous),
            "removed": sorted(previous - discovered),
        }

    def snapshot(self) -> set[str]:
        with self._lock:
            return set(self._urls)

    @staticmethod
    def _validate_url(value: str) -> str:
        normalized = value.rstrip("/")
        parsed = urlparse(normalized)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid training service URL: {value}")
        return normalized


def next_poll_delay(schedule: str) -> float:
    now = datetime.now(timezone.utc)
    next_poll = croniter(schedule, now).get_next(datetime)
    return max(0.0, (next_poll - now).total_seconds())
