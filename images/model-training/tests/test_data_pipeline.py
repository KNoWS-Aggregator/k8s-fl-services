import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import model_training.data_pipeline as pipeline
from model_training.data_pipeline import (
    ACTIVITIES,
    WINDOW_SAMPLES,
    _load_or_create_assignments,
    _align_labels,
    load_shared_dataset,
)


def test_align_labels_normalizes_timestamp_units():
    window_times = np.array(["2026-01-01T00:00:00.500000000"], dtype="datetime64[ns]")
    gt = pd.DataFrame(
        {"GT": ["Walking"]},
        index=pd.DatetimeIndex(
            np.array(["2026-01-01T00:00:00.500000"], dtype="datetime64[us]"),
            name="event_time",
        ),
    )

    labels = _align_labels(window_times, gt)

    assert labels.tolist() == ["Walking"]


def test_shared_parquet_is_windowed_per_participant(monkeypatch, tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    timestamps = np.arange(
        np.datetime64("2026-01-01T00:00:00"),
        np.datetime64("2026-01-01T00:00:07"),
        np.timedelta64(31250000, "ns"),
    )
    participants = ["participant-a", "participant-b"]
    accel_rows = {
        "participant_id": [],
        "event_time": [],
        "ACC_x": [],
        "ACC_y": [],
        "ACC_z": [],
    }
    gt_rows = {"participant_id": [], "event_time": [], "GT": []}
    for index, participant in enumerate(participants):
        count = len(timestamps)
        accel_rows["participant_id"].extend([participant] * count)
        accel_rows["event_time"].extend(timestamps)
        accel_rows["ACC_x"].extend(np.full(count, 1.0 + index))
        accel_rows["ACC_y"].extend(np.full(count, 2.0 + index))
        # Avoid triggering the OnTable override.
        accel_rows["ACC_z"].extend(np.full(count, 5.0))
        gt_rows["participant_id"].append(participant)
        gt_rows["event_time"].append(timestamps[0])
        gt_rows["GT"].append("Walking" if index == 0 else "Sitting")
    pq.write_table(pa.table(accel_rows), downloads / "accel.parquet")
    pq.write_table(pa.table(gt_rows), downloads / "gt.parquet")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    dataset = load_shared_dataset()

    assert dataset.inputs.shape == (2, WINDOW_SAMPLES, 3)
    assert set(dataset.participant_ids) == set(participants)
    assert np.allclose(dataset.inputs[0, 0], np.array([1.0, 2.0, 5.0]) / 9.81)
    labels = np.argmax(dataset.labels, axis=1)
    assert set(labels) == {ACTIVITIES.index("Walking"), ACTIVITIES.index("Sitting")}

    monkeypatch.setattr(
        pipeline,
        "_process_participant",
        lambda *_args, **_kwargs: pytest.fail("cache was not reused"),
    )
    cached = load_shared_dataset()
    assert np.array_equal(cached.inputs, dataset.inputs)


def test_participant_split_registry_is_stable(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    initial = _load_or_create_assignments(
        ["p1", "p2", "p3", "p4", "p5"],
        val_size=0.2,
        test_size=0.2,
    )

    assert list(initial.values()).count("train") == 3
    assert list(initial.values()).count("val") == 1
    assert list(initial.values()).count("test") == 1

    updated = _load_or_create_assignments(
        ["p2", "p3", "p4", "p5", "p6"],
        val_size=0.2,
        test_size=0.2,
    )

    for participant in {"p2", "p3", "p4", "p5"}:
        assert updated[participant] == initial[participant]
    assert "p1" not in updated
    assert "p6" in updated

    with pytest.raises(ValueError, match="Split configuration differs"):
        _load_or_create_assignments(
            ["p2", "p3", "p4", "p5", "p6"],
            val_size=0.4,
            test_size=0.2,
        )
