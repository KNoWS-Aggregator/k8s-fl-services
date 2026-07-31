from fastapi import BackgroundTasks
import numpy as np
import pytest

from common.messages import (
    EvaluationMetrics,
    EvaluationResultMessage,
    TrainingMetrics,
    TrainingResultMessage,
)
from common.weight_io import bytes_to_weights, weights_to_bytes
from fl_model import weight_signature
from weight_aggregation import main


def _run_tasks(tasks):
    for task in tasks.tasks:
        task.func(*task.args, **task.kwargs)


def test_session_bootstrap_aggregates_and_promotes_global_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        main,
        "TRAINING_CLIENT_BASE_URLS",
        ["http://training-a", "http://training-b"],
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

    for loss, accuracy, client_id in zip((0.8, 0.4), (0.2, 0.8), client_ids):
        main._record_evaluation_result(
            EvaluationResultMessage(
                session_id=session.session_id,
                round_id=1,
                client_id=client_id,
                metrics=EvaluationMetrics(
                    num_examples=1,
                    eval_loss=loss,
                    eval_accuracy=accuracy,
                ),
            )
        )

    assert main._active is None
    promoted = bytes_to_weights(main._global_weights_path().read_bytes())
    assert np.allclose(promoted[0], np.array([3.0]))
    evaluation = main.aggregated_evaluation_metrics()
    assert evaluation["metrics"]["num_examples"] == 2
    assert evaluation["metrics"]["eval_loss"] == pytest.approx(0.6)
    assert evaluation["metrics"]["eval_accuracy"] == 0.5
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
