"""In-memory tracking of the current training round: who we're waiting
on, and what they've reported so far."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from common.messages import TrainingMetrics


@dataclass
class RoundState:
    round_id: int
    expected_participants: set[str]
    weights_by_participant: dict[str, bytes] = field(default_factory=dict)
    metrics_by_participant: dict[str, TrainingMetrics] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.expected_participants.issubset(self.weights_by_participant.keys())


class RoundTracker:
    """Thread-safe holder for the single in-progress round. Only one
    round is tracked at a time - starting a new one replaces the old."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: RoundState | None = None

    def start_round(self, round_id: int, participant_ids: set[str]) -> None:
        with self._lock:
            self._current = RoundState(round_id=round_id, expected_participants=participant_ids)

    def record_result(self, round_id: int, participant_id: str,
                       weights: bytes, metrics: TrainingMetrics) -> RoundState:
        """Records one participant's result. Raises if it doesn't match
        the currently tracked round (stale or misdirected message)."""
        with self._lock:
            state = self._current
            if state is None or state.round_id != round_id:
                raise ValueError(
                    f"Received result for round {round_id}, "
                    f"but current round is {state.round_id if state else None}"
                )
            if participant_id not in state.expected_participants:
                raise ValueError(f"Unexpected participant '{participant_id}' for round {round_id}")

            state.weights_by_participant[participant_id] = weights
            state.metrics_by_participant[participant_id] = metrics
            return state

    def current(self) -> RoundState | None:
        with self._lock:
            return self._current


tracker = RoundTracker()