import asyncio
import json

import pytest
from fastapi import BackgroundTasks, HTTPException

from common.messages import (
    ClientSessionStart,
    EvaluationInitMessage,
    EvaluationMetrics,
    EvaluationResultMessage,
    TrainingConfig,
    TrainingFailureMessage,
    TrainingInitMessage,
    TrainingMetrics,
    TrainingResultMessage,
)
from model_training import main


def _message(round_id=1):
    return TrainingInitMessage(
        session_id="session-1",
        round_id=round_id,
        client_id="hospital-a",
        reply_url="http://weight-aggregation/results",
    )


class _Upload:
    async def read(self):
        return b"global-weights"


def _request(message):
    if main._read_session() is None:
        main._write_session(
            ClientSessionStart(
                session_id=message.session_id,
                client_id=message.client_id,
                expected_rounds=10,
                training_config=TrainingConfig(),
            )
        )
    tasks = BackgroundTasks()
    response = asyncio.run(
        main.train_endpoint(
            background_tasks=tasks,
            message=message.model_dump_json(),
            weights=_Upload(),
        )
    )
    return response, tasks


def _evaluation_message(round_id=1):
    return EvaluationInitMessage(
        session_id="session-1",
        round_id=round_id,
        client_id="hospital-a",
        reply_url="http://weight-aggregation/evaluation-results",
    )


def _evaluation_request(message):
    if main._read_session() is None:
        main._write_session(
            ClientSessionStart(
                session_id=message.session_id,
                client_id=message.client_id,
                expected_rounds=10,
                training_config=TrainingConfig(),
            )
        )
    tasks = BackgroundTasks()
    response = asyncio.run(
        main.evaluate_endpoint(
            background_tasks=tasks,
            message=message.model_dump_json(),
            weights=_Upload(),
        )
    )
    return response, tasks


def _run_tasks(tasks):
    for task in tasks.tasks:
        task.func(*task.args, **task.kwargs)


def test_completed_round_is_cached_and_duplicate_is_re_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    calls = {"train": 0, "reports": []}

    def fake_training(message, _config, _weights):
        calls["train"] += 1
        return (
            TrainingResultMessage(
                session_id=message.session_id,
                round_id=message.round_id,
                client_id=message.client_id,
                metrics=TrainingMetrics(num_examples=12),
            ),
            b"local-weights",
        )

    monkeypatch.setattr(main, "run_training", fake_training)
    monkeypatch.setattr(
        main,
        "post_message",
        lambda url, message, weights=None: calls["reports"].append((url, message, weights)),
    )

    accepted, tasks = _request(_message())
    assert accepted["status"] == "accepted"
    _run_tasks(tasks)
    assert main.training_status()["status"] == "succeeded"

    duplicate, duplicate_tasks = _request(_message())
    assert duplicate["status"] == "already_completed"
    _run_tasks(duplicate_tasks)

    assert calls["train"] == 1
    assert len(calls["reports"]) == 2
    assert calls["reports"][1][2] == b"local-weights"


def test_different_concurrent_round_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert main._run_lock.acquire(blocking=False)
    try:
        main._write_status(
            {"status": "running", "round_id": 1, "client_id": "hospital-a"}
        )
        same, _ = _request(_message(round_id=1))
        assert same["status"] == "already_running"
        with pytest.raises(HTTPException) as error:
            _request(_message(round_id=2))
        assert error.value.status_code == 409
    finally:
        main._run_lock.release()


def test_training_failure_is_persisted_and_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    reports = []

    def fail_training(_message, _config, _weights):
        raise RuntimeError("training exploded")

    monkeypatch.setattr(main, "run_training", fail_training)
    monkeypatch.setattr(
        main,
        "post_message",
        lambda url, message, weights=None: reports.append((url, message, weights)),
    )

    _, tasks = _request(_message())
    _run_tasks(tasks)

    assert main.training_status()["status"] == "failed"
    assert isinstance(reports[0][1], TrainingFailureMessage)
    assert reports[0][1].error == "training exploded"


def test_evaluation_is_cached_reported_and_exposed(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    calls = {"evaluate": 0, "reports": []}

    def fake_evaluation(message, _config, _weights):
        calls["evaluate"] += 1
        return EvaluationResultMessage(
            session_id=message.session_id,
            round_id=message.round_id,
            client_id=message.client_id,
            metrics=EvaluationMetrics(
                num_examples=8,
                eval_loss=0.25,
                eval_accuracy=0.75,
            ),
        )

    monkeypatch.setattr(main, "run_evaluation", fake_evaluation)
    monkeypatch.setattr(
        main,
        "post_message",
        lambda url, message, weights=None: calls["reports"].append((url, message, weights)),
    )

    accepted, tasks = _evaluation_request(_evaluation_message())
    assert accepted["status"] == "accepted"
    _run_tasks(tasks)
    assert main.training_status()["status"] == "evaluation_succeeded"

    duplicate, duplicate_tasks = _evaluation_request(_evaluation_message())
    assert duplicate["status"] == "already_completed"
    _run_tasks(duplicate_tasks)

    exposed = main.training_metrics()
    assert exposed["training"] is None
    assert exposed["evaluation"]["num_examples"] == 8
    assert exposed["evaluation"]["eval_accuracy"] == 0.75
    assert calls["evaluate"] == 1
    assert len(calls["reports"]) == 2


def test_readiness_checks_shared_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    with pytest.raises(HTTPException) as error:
        main.readiness()
    assert error.value.status_code == 503

    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "accel.parquet").touch()
    (downloads / "gt.parquet").touch()
    (tmp_path / "dataset-state.json").write_text(
        json.dumps({"status": "updating", "generation": "new"}),
        encoding="utf-8",
    )
    with pytest.raises(HTTPException) as updating:
        main.readiness()
    assert updating.value.status_code == 503

    (tmp_path / "dataset-state.json").write_text(
        json.dumps({"status": "ready", "generation": "new"}),
        encoding="utf-8",
    )
    assert main.readiness() == {"status": "ready"}


def test_history_and_weights_include_active_round(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    completed = tmp_path / "model-training" / "rounds" / "session-1" / "hospital-a" / "round_1"
    active = tmp_path / "model-training" / "rounds" / "session-1" / "hospital-a" / "round_2"
    completed.mkdir(parents=True)
    active.mkdir(parents=True)
    (completed / "history.json").write_text(json.dumps([{"loss": 0.5}]), encoding="utf-8")
    (completed / "weights.npz").write_bytes(b"round-1")
    (completed / "result.json").write_text(
        TrainingResultMessage(
            session_id="session-1",
            round_id=1,
            client_id="hospital-a",
            metrics=TrainingMetrics(num_examples=12),
        ).model_dump_json(),
        encoding="utf-8",
    )
    training_only = main.training_metrics()
    assert training_only["training"]["num_examples"] == 12
    assert training_only["evaluation"] is None
    (completed / "evaluation.json").write_text(
        EvaluationResultMessage(
            session_id="session-1",
            round_id=1,
            client_id="hospital-a",
            metrics=EvaluationMetrics(num_examples=4, eval_accuracy=0.75),
        ).model_dump_json(),
        encoding="utf-8",
    )
    (active / "history.json").write_text(json.dumps([{"loss": 0.4}]), encoding="utf-8")
    (active / "weights.npz").write_bytes(b"round-2")
    main._write_status(
        {"status": "running", "session_id": "session-1", "client_id": "hospital-a", "round_id": 2}
    )

    history = main.training_history()
    assert [round_data["round_id"] for round_data in history["rounds"]] == [1, 2]
    assert history["rounds"][1]["history"] == [{"loss": 0.4}]
    metrics = main.training_metrics()
    assert metrics["round_id"] == 1
    assert metrics["training"]["num_examples"] == 12
    assert metrics["evaluation"]["num_examples"] == 4
    assert metrics["evaluation"]["eval_accuracy"] == 0.75
    assert main.training_weights().path == active / "weights.npz"


def test_updating_dataset_is_trainable_when_previous_generation_exists(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "accel.parquet").touch()
    (downloads / "gt.parquet").touch()
    (tmp_path / "dataset-state.json").write_text(
        json.dumps({"status": "updating", "generation": "next"}),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"generation": "previous", "participants": {}}),
        encoding="utf-8",
    )

    status_payload = main.training_status()

    assert status_payload["prepared_data"] == {
        "valid": True,
        "status": "updating",
        "generation": "previous",
    }
    assert status_payload["session"] == {"active": False, "session_id": None}
    assert status_payload["trainable"] is True
    assert main.readiness() == {"status": "ready"}


def test_training_client_session_assignment(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        main,
        "_prepared_data_status",
        lambda: {"valid": True, "status": "ready", "generation": "test"},
    )
    monkeypatch.setattr(main, "create_initial_weights", lambda: ([], "sha256:model"))
    request = ClientSessionStart(
        session_id="session-a",
        client_id="client-a",
        expected_rounds=2,
        training_config=TrainingConfig(local_epochs=3),
    )

    started = main.session_start(request)
    assert started["status"] == "started"
    assert main._read_session() == request
    assert main.session_start(request)["status"] == "already_started"

    with pytest.raises(HTTPException) as conflict:
        main.session_start(
            ClientSessionStart(
                session_id="session-b",
                client_id="client-b",
                expected_rounds=1,
            )
        )
    assert conflict.value.status_code == 409

    assert main.session_end(
        main.ClientSessionEnd(session_id="session-a", client_id="client-a")
    ) == {"status": "ended"}
    assert main._read_session() is None
