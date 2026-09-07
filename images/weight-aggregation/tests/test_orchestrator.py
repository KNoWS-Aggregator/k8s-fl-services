from fastapi import BackgroundTasks, HTTPException
import numpy as np
import pytest
from unittest.mock import Mock

from common.messages import (
    EvaluationMetrics,
    EvaluationResultMessage,
    TrainingMetrics,
    TrainingResultMessage,
)
from common.weight_io import bytes_to_weights, weights_to_bytes
from fl_model import weight_signature
from weight_aggregation import main


class FakeRegistry:
    def __init__(self, urls):
        self.urls = set(urls)

    def snapshot(self):
        return set(self.urls)


def _run_tasks(tasks):
    for task in tasks.tasks:
        task.func(*task.args, **task.kwargs)


def test_global_weights_download_serves_only_promoted_model(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    with pytest.raises(HTTPException) as missing:
        main.global_weights()
    assert missing.value.status_code == 404

    payload = weights_to_bytes([np.array([1.0, 2.0])])
    path = main._global_weights_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    response = main.global_weights()

    assert response.path == path
    assert response.media_type == "application/octet-stream"
    assert response.headers["content-disposition"] == 'attachment; filename="global-weights.npz"'


def test_session_bootstrap_aggregates_and_promotes_global_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        main, "_registry", FakeRegistry(["http://training-a", "http://training-b"])
    )
    initial = weights_to_bytes([np.array([1.0])])
    signature = weight_signature(bytes_to_weights(initial))
    load_persisted_weights = main._initial_global_weights
    monkeypatch.setattr(
        main,
        "_initial_global_weights",
        lambda: (initial, signature, "fresh_model"),
    )
    monkeypatch.setattr(
        main,
        "_client_post",
        lambda base, path, message: {"model_signature": signature},
    )
    dispatched = {}

    def fake_dispatch(**kwargs):
        dispatched.update(kwargs)
        return set()

    monkeypatch.setattr(main, "dispatch_round", fake_dispatch)
    monkeypatch.setattr(main, "dispatch_evaluation", lambda **_kwargs: set())
    monkeypatch.setattr(
        main,
        "_client_counts",
        lambda: (
            2,
            2,
            {"http://training-a", "http://training-b"},
        ),
    )

    tasks = BackgroundTasks()
    response = main.session_start(
        main.SessionStartRequest(expected_rounds=1),
        tasks,
    )
    _run_tasks(tasks)

    session = main._active
    assert response["status"] == "initializing"
    assert session is not None
    assert session.current_round == 1
    assert len(session.active_clients) == 2
    assert len(set(session.active_clients)) == 2
    assert dispatched["round_id"] == 1
    if session.timer:
        session.timer.cancel()

    client_ids = list(session.active_clients)
    for value, client_id in zip((2.0, 4.0), client_ids):
        local = weights_to_bytes([np.array([value])])
        main._record_result(
            TrainingResultMessage(
                session_id=session.session_id,
                round_id=1,
                client_id=client_id,
                metrics=TrainingMetrics(num_examples=1),
            ),
            local,
        )

    matrices = ([[1, 0], [0, 0]], [[0, 0], [1, 0]])
    per_class = {"Walking": {}, "Sitting": {}}
    for loss, accuracy, matrix, client_id in zip((0.8, 0.4), (0.2, 0.8), matrices, client_ids):
        main._record_evaluation_result(
            EvaluationResultMessage(
                session_id=session.session_id,
                round_id=1,
                client_id=client_id,
                metrics=EvaluationMetrics(
                    num_examples=1,
                    eval_loss=loss,
                    eval_accuracy=accuracy,
                    confusion_matrix=matrix,
                    per_class=per_class,
                ),
            )
        )

    assert main._active is None
    promoted = bytes_to_weights(main._global_weights_path().read_bytes())
    assert np.allclose(promoted[0], np.array([3.0]))
    evaluation = main.metrics()["evaluation"]
    assert evaluation["num_examples"] == 2
    assert evaluation["eval_loss"] == pytest.approx(0.6)
    assert evaluation["eval_accuracy"] == 0.5
    assert evaluation["confusion_matrix"] == [[1, 0], [1, 0]]
    assert evaluation["eval_f1_macro"] == pytest.approx(1 / 3)
    assert evaluation["eval_recall_macro"] == 0.5
    payload, persisted_signature, source = load_persisted_weights()
    assert payload == main._global_weights_path().read_bytes()
    assert persisted_signature == signature
    assert source == "previous_session"


def test_failed_client_is_removed_while_remaining_clients_continue(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "_client_post", lambda *_args, **_kwargs: {})
    weights = weights_to_bytes([np.array([1.0])])
    signature = weight_signature(bytes_to_weights(weights))
    session = main.ActiveSession(
        session_id="session-1",
        request=main.SessionStartRequest(expected_rounds=1, min_clients=1),
        all_clients={"a": "http://a", "b": "http://b", "c": "http://c"},
        active_clients={"a": "http://a", "b": "http://b", "c": "http://c"},
        model_signature=signature,
        current_round=1,
        global_weights=weights,
        expected_clients={"a", "b", "c"},
    )
    main._active = session
    monkeypatch.setattr(main, "dispatch_evaluation", lambda **_kwargs: set())
    monkeypatch.setattr(main, "_client_counts", lambda: (3, 3, {"http://a", "http://b", "http://c"}))

    main._record_result(
        TrainingResultMessage(
            session_id="session-1",
            round_id=1,
            client_id="a",
            metrics=TrainingMetrics(num_examples=1),
        ),
        weights_to_bytes([np.array([2.0])]),
    )
    main._record_failure("session-1", 1, "b", "client failed")
    main._record_result(
        TrainingResultMessage(
            session_id="session-1",
            round_id=1,
            client_id="c",
            metrics=TrainingMetrics(num_examples=3),
        ),
        weights_to_bytes([np.array([4.0])]),
    )

    for client_id, examples in (("a", 1), ("c", 3)):
        main._record_evaluation_result(
            EvaluationResultMessage(
                session_id="session-1",
                round_id=1,
                client_id=client_id,
                metrics=EvaluationMetrics(
                    num_examples=examples,
                    eval_loss=0.5,
                    eval_accuracy=0.75,
                ),
            )
        )

    assert main._active is None
    promoted = bytes_to_weights(main._global_weights_path().read_bytes())
    assert np.allclose(promoted[0], np.array([3.5]))
    assert main.session_status()["status"] == "succeeded"


def test_registry_changes_are_applied_only_when_next_round_starts(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    registry = FakeRegistry(["http://training-a"])
    monkeypatch.setattr(main, "_registry", registry)
    weights = weights_to_bytes([np.array([1.0])])
    signature = weight_signature(bytes_to_weights(weights))
    session = main.ActiveSession(
        session_id="session-membership",
        request=main.SessionStartRequest(expected_rounds=2, min_clients=1),
        all_clients={"a": "http://training-a"},
        active_clients={"a": "http://training-a"},
        model_signature=signature,
        current_round=1,
        global_weights=weights,
        expected_clients={"a"},
    )
    main._active = session
    monkeypatch.setattr(
        main,
        "_client_post",
        lambda _base, path, _message: (
            {"model_signature": signature} if path == main.CLIENT_SESSION_START_PATH else {}
        ),
    )
    dispatched = {}
    monkeypatch.setattr(main, "dispatch_round", lambda **kwargs: dispatched.update(kwargs) or set())

    registry.urls = {"http://training-b"}
    assert session.expected_clients == {"a"}
    assert session.active_clients == {"a": "http://training-a"}

    main._start_round(session)

    assert session.current_round == 2
    assert set(session.active_clients.values()) == {"http://training-b"}
    assert session.expected_clients == set(session.active_clients)
    assert set(dispatched["client_urls"].values()) == {"http://training-b/train"}
    if session.timer:
        session.timer.cancel()


def test_client_trainability_requires_data_and_no_foreign_session(monkeypatch):
    payload = {
        "prepared_data": {"valid": True},
        "session": {"active": True, "session_id": "session-a"},
    }
    monkeypatch.setattr(
        main,
        "send_request",
        lambda *_args, **_kwargs: Mock(
            raise_for_status=lambda: None,
            json=lambda: payload,
        ),
    )

    assert main._is_trainable("http://training-a", "session-a") is True
    assert main._is_trainable("http://training-a", "session-b") is False
    assert main._is_trainable("http://training-a", None) is False

    payload["prepared_data"]["valid"] = False
    payload["session"] = {"active": False, "session_id": None}
    assert main._is_trainable("http://training-a", None) is False


def test_status_reports_registered_and_trainable_client_counts(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        main,
        "_registry",
        FakeRegistry(["http://training-a", "http://training-b", "http://training-c"]),
    )
    monkeypatch.setattr(
        main,
        "_is_trainable",
        lambda url, _session_id: url == "http://training-a",
    )
    main._active = None

    payload = main.session_status()

    assert payload["status"] == "idle"
    assert payload["registered_clients"] == 3
    assert payload["trainable_clients"] == 1


def test_metrics_return_latest_round_and_history(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "_client_counts", lambda: (2, 1, set()))
    payload = {
        "session_id": "session-1",
        "rounds": [
            {
                "round_id": 1,
                "client_training": {"client-a": {"train_loss": 0.5}},
                "evaluation_metrics": {"eval_accuracy": 0.75},
                "server": {"aggregation_seconds": 0.25},
            }
        ],
    }
    main._write_json(main._round_metrics_path(), payload)

    assert "round_metrics" not in main.session_status()
    latest = main.metrics()
    assert latest["session_id"] == "session-1"
    assert latest["round_id"] == 1
    assert latest["training"]["client-a"]["train_loss"] == 0.5
    assert latest["evaluation"]["eval_accuracy"] == 0.75
    assert main.metrics_history()["rounds"] == [
        {
            "round_id": 1,
            "training": {"client-a": {"train_loss": 0.5}},
            "evaluation": {"eval_accuracy": 0.75},
            "server": {"aggregation_seconds": 0.25},
        }
    ]


def test_metrics_include_active_round_with_pending_evaluation(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    session = main.ActiveSession(
        session_id="session-active",
        request=main.SessionStartRequest(expected_rounds=2),
        all_clients={},
        active_clients={},
        model_signature="signature",
        current_round=2,
        round_metrics={
            "2": {
                "round_id": 2,
                "client_training": {"client-a": {"train_loss": 0.4}},
                "server": {"aggregation_seconds": 0.1},
            }
        },
    )
    monkeypatch.setattr(main, "_active", session)

    payload = main.metrics()

    assert payload["session_id"] == "session-active"
    assert payload["round_id"] == 2
    assert payload["training"]["client-a"]["train_loss"] == 0.4
    assert payload["evaluation"] is None


def test_refresh_clients_immediately_refreshes_registry(monkeypatch):
    changes = {
        "added": ["http://training-new"],
        "removed": ["http://training-old"],
    }
    refresh = Mock(return_value=changes)
    monkeypatch.setattr(main, "_registry", Mock(refresh=refresh))

    assert main.refresh_clients() == changes
    refresh.assert_called_once_with()
