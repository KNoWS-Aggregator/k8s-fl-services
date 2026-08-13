"""Transform data-preparation Parquet outputs into CNN-ready local arrays.

Each participant is processed independently. Only completed windows are
concatenated, ensuring that a model window can never cross participant
boundaries.
"""
from __future__ import annotations

import os
import json
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from tsflex.features import FeatureCollection, FeatureDescriptor
from tsflex.features.utils import make_robust

ACTIVITIES = ["Lying", "OnTable", "Sitting", "Standing", "Walking"]
WINDOW_SECONDS = 6
SAMPLING_FREQUENCY = 32
WINDOW_OVERLAP = 0.5
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLING_FREQUENCY
WINDOW_STEP = int(WINDOW_SAMPLES * WINDOW_OVERLAP)
PROCESSING_VERSION = 1


@dataclass(frozen=True)
class ModelDataset:
    inputs: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    participant_ids: np.ndarray

    @property
    def num_examples(self) -> int:
        return len(self.inputs)


def _paths_from_env() -> tuple[Path, Path]:
    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))
    accel_path = Path(os.getenv("ACCEL_PARQUET_PATH", str(data_dir / "downloads" / "accel.parquet")))
    gt_path = Path(os.getenv("GT_PARQUET_PATH", str(data_dir / "downloads" / "gt.parquet")))
    for name, path in (("accelerometer", accel_path), ("ground-truth", gt_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Shared {name} Parquet file not found: {path}")
    return accel_path, gt_path


def _fraction(value: float, name: str) -> float:
    fraction = value / 100 if value > 1 else value
    if not 0 <= fraction < 1:
        raise ValueError(f"{name} must be a fraction from 0 to <1 or a percentage from 0 to <100")
    return fraction


def _split_registry_path() -> Path:
    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))
    return Path(
        os.getenv(
            "SPLIT_REGISTRY_PATH",
            str(data_dir / "model-training" / "participant-splits.json"),
        )
    )


def _dataset_state_path() -> Path:
    return Path(
        os.getenv(
            "DATASET_STATE_PATH",
            str(Path(os.getenv("DATA_DIR", "/app/data")) / "dataset-state.json"),
        )
    )


def _dataset_generation(accel_path: Path, gt_path: Path) -> str:
    state_path = _dataset_state_path()
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "ready" and state.get("generation"):
            return str(state["generation"])
        manifest_path = state_path.with_name("manifest.json")
        if manifest_path.is_file() and accel_path.is_file() and gt_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("generation"):
                available = [
                    (str(path), path.stat().st_size, path.stat().st_mtime_ns)
                    for path in (accel_path, gt_path)
                ]
                fingerprint = hashlib.sha256(
                    json.dumps(available, separators=(",", ":")).encode()
                ).hexdigest()
                return f"{manifest['generation']}-{fingerprint}"
        raise RuntimeError(
            f"Prepared dataset is not ready: {state.get('status', 'unknown')}"
        )
    # Compatibility for volumes produced before the generation handshake.
    legacy = [
        (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in (accel_path, gt_path)
    ]
    return "legacy-" + hashlib.sha256(
        json.dumps(legacy, separators=(",", ":")).encode()
    ).hexdigest()


def _cache_dir() -> Path:
    return Path(
        os.getenv(
            "MODEL_DATA_CACHE_DIR",
            str(Path(os.getenv("DATA_DIR", "/app/data")) / "model-training" / "cache"),
        )
    )


def _target_counts(total: int, val_fraction: float, test_fraction: float) -> dict[str, int]:
    validation = round(total * val_fraction)
    test = round(total * test_fraction)
    requested_holdouts = int(val_fraction > 0) + int(test_fraction > 0)
    if total > requested_holdouts:
        if val_fraction > 0:
            validation = max(1, validation)
        if test_fraction > 0:
            test = max(1, test)
    while validation + test >= total and validation + test:
        if test >= validation and test:
            test -= 1
        elif validation:
            validation -= 1
    return {"train": total - validation - test, "val": validation, "test": test}


def _load_or_create_assignments(
    participant_ids: list[str],
    val_size: float,
    test_size: float,
) -> dict[str, str]:
    val_fraction = _fraction(val_size, "val_size")
    test_fraction = _fraction(test_size, "test_size")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_size and test_size must leave a non-zero training fraction")
    seed = int(os.getenv("SPLIT_SEED", "42"))
    registry_path = _split_registry_path()
    configuration = {
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
    }
    assignments: dict[str, str] = {}
    if registry_path.is_file():
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        if payload.get("configuration") != configuration:
            raise ValueError(
                f"Split configuration differs from persistent registry {registry_path}; "
                "use the original val_size/test_size/SPLIT_SEED or remove the registry intentionally"
            )
        stored = payload.get("assignments", {})
        assignments = {
            participant: split
            for participant, split in stored.items()
            if participant in participant_ids and split in {"train", "val", "test"}
        }

    new_participants = sorted(set(participant_ids) - set(assignments))
    random.Random(seed).shuffle(new_participants)
    targets = _target_counts(len(participant_ids), val_fraction, test_fraction)
    counts = {
        split: sum(assigned == split for assigned in assignments.values())
        for split in ("train", "val", "test")
    }
    for participant in new_participants:
        split = max(
            ("train", "val", "test"),
            key=lambda candidate: (targets[candidate] - counts[candidate], candidate == "train"),
        )
        assignments[participant] = split
        counts[split] += 1

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = registry_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"configuration": configuration, "assignments": dict(sorted(assignments.items()))},
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(registry_path)
    return assignments


def _read_participant_frames(
    connection: duckdb.DuckDBPyConnection,
    accel_path: Path,
    gt_path: Path,
    participant_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accel = connection.execute(
        """
        SELECT event_time, ACC_x, ACC_y, ACC_z
        FROM read_parquet(?)
        WHERE participant_id = ?
        ORDER BY event_time
        """,
        [str(accel_path), participant_id],
    ).fetchdf()
    gt = connection.execute(
        """
        SELECT event_time, GT
        FROM read_parquet(?)
        WHERE participant_id = ?
        ORDER BY event_time
        """,
        [str(gt_path), participant_id],
    ).fetchdf()
    accel = accel.set_index(pd.DatetimeIndex(accel.pop("event_time"), name="event_time"))
    gt = gt.set_index(pd.DatetimeIndex(gt.pop("event_time"), name="event_time"))
    return accel, gt


def _phone_on_table(accel: pd.DataFrame) -> pd.DataFrame:
    descriptors = []
    for function in (np.mean, np.std):
        descriptors.extend(
            FeatureDescriptor(
                function=make_robust(function),
                series_name=axis,
                window="1s",
                stride="0.5s",
            )
            for axis in ("ACC_x", "ACC_y", "ACC_z")
        )
    features = FeatureCollection(feature_descriptors=descriptors).calculate(
        accel.dropna(how="all"),
        return_df=True,
        approve_sparsity=True,
        n_jobs=1,
    )
    rename = {}
    for column in features.columns:
        text = str(column)
        for axis in ("ACC_x", "ACC_y", "ACC_z"):
            if text.startswith(f"{axis}__mean"):
                rename[column] = f"{axis}_mean"
            elif text.startswith(f"{axis}__std"):
                rename[column] = f"{axis}_std"
    features = features.rename(columns=rename)
    required = {
        "ACC_x_mean", "ACC_y_mean", "ACC_z_mean",
        "ACC_x_std", "ACC_y_std", "ACC_z_std",
    }
    if not required.issubset(features.columns):
        missing = ", ".join(sorted(required - set(features.columns)))
        raise RuntimeError(f"Phone-on-table feature extraction did not produce: {missing}")
    features["is_on_table"] = (
        (features["ACC_x_std"] < 0.7)
        & (features["ACC_y_std"] < 0.7)
        & (features["ACC_z_std"] < 0.7)
        & ((features["ACC_z_mean"] > 9) | (features["ACC_z_mean"] < -9))
    )
    return features[["is_on_table"]]


def _override_on_table(gt: pd.DataFrame, on_table: pd.DataFrame) -> pd.DataFrame:
    result = gt.copy()
    if result.empty or on_table.empty:
        return result
    merged = pd.merge_asof(
        result.dropna(subset=["GT"]).reset_index().sort_values("event_time"),
        on_table.reset_index().sort_values("event_time"),
        on="event_time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=1),
    )
    timestamps = merged.loc[merged["is_on_table"].fillna(False), "event_time"]
    result.loc[result.index.isin(timestamps), "GT"] = "OnTable"
    return result


def _rolling_windows(accel_g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = accel_g[["ACC_x", "ACC_y", "ACC_z"]].to_numpy(dtype=np.float32)
    timestamps = accel_g.index.to_numpy()
    windows = []
    starts = []
    for start in range(0, len(values), WINDOW_STEP):
        window = values[start:start + WINDOW_SAMPLES]
        if len(window) == WINDOW_SAMPLES:
            windows.append(window)
            starts.append(timestamps[start])
    if not windows:
        return (
            np.empty((0, WINDOW_SAMPLES, 3), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )
    return np.stack(windows), np.asarray(starts, dtype="datetime64[ns]")


def _align_labels(window_times: np.ndarray, gt: pd.DataFrame) -> np.ndarray:
    if not len(window_times) or gt.empty:
        return np.full(len(window_times), None, dtype=object)
    windows = pd.DataFrame({"event_time": pd.to_datetime(window_times)}).sort_values("event_time")
    labels = gt.dropna(subset=["GT"]).reset_index().sort_values("event_time")
    # Parquet/Arrow commonly yields datetime64[us], while rolling-window
    # timestamps are explicitly datetime64[ns]. pandas.merge_asof requires the
    # join keys to have exactly the same dtype, even though both represent the
    # same (timezone-naive) timestamps.
    windows["event_time"] = windows["event_time"].astype("datetime64[ns]")
    labels["event_time"] = labels["event_time"].astype("datetime64[ns]")
    aligned = pd.merge_asof(
        windows,
        labels,
        on="event_time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=WINDOW_SECONDS),
    )
    return aligned["GT"].to_numpy(dtype=object)


def _process_participant(
    participant_id: str,
    accel: pd.DataFrame,
    gt: pd.DataFrame,
) -> ModelDataset:
    if accel.empty or gt.empty:
        return _empty_dataset()
    on_table = _phone_on_table(accel)
    gt = _override_on_table(gt, on_table)
    accel_g = accel.copy()
    accel_g.loc[:, ["ACC_x", "ACC_y", "ACC_z"]] /= 9.81
    inputs, timestamps = _rolling_windows(accel_g)
    labels = _align_labels(timestamps, gt)
    valid = np.isin(labels, ACTIVITIES) & ~np.isnan(inputs).any(axis=(1, 2))
    inputs, timestamps, labels = inputs[valid], timestamps[valid], labels[valid]
    if not len(inputs):
        return _empty_dataset()
    indices = np.asarray([ACTIVITIES.index(label) for label in labels])
    encoded = to_categorical(indices, num_classes=len(ACTIVITIES)).astype(np.float32)
    return ModelDataset(
        inputs=inputs,
        labels=encoded,
        timestamps=timestamps,
        participant_ids=np.full(len(inputs), participant_id, dtype=object),
    )


def _empty_dataset() -> ModelDataset:
    return ModelDataset(
        inputs=np.empty((0, WINDOW_SAMPLES, 3), dtype=np.float32),
        labels=np.empty((0, len(ACTIVITIES)), dtype=np.float32),
        timestamps=np.empty((0,), dtype="datetime64[ns]"),
        participant_ids=np.empty((0,), dtype=object),
    )


def _combine(datasets: list[ModelDataset]) -> ModelDataset:
    if not datasets:
        return _empty_dataset()
    return ModelDataset(
        inputs=np.concatenate([dataset.inputs for dataset in datasets]),
        labels=np.concatenate([dataset.labels for dataset in datasets]),
        timestamps=np.concatenate([dataset.timestamps for dataset in datasets]),
        participant_ids=np.concatenate([dataset.participant_ids for dataset in datasets]),
    )


def _cache_key(
    generation: str,
    assignments: dict[str, str],
    val_size: float,
    test_size: float,
) -> str:
    payload = {
        "generation": generation,
        "assignments": assignments,
        "val_fraction": _fraction(val_size, "val_size"),
        "test_fraction": _fraction(test_size, "test_size"),
        "split_seed": int(os.getenv("SPLIT_SEED", "42")),
        "processing_version": PROCESSING_VERSION,
        "activities": ACTIVITIES,
        "window_seconds": WINDOW_SECONDS,
        "sampling_frequency": SAMPLING_FREQUENCY,
        "window_overlap": WINDOW_OVERLAP,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_cache(path: Path) -> dict[str, ModelDataset]:
    with np.load(path, allow_pickle=True) as cached:
        return {
            split: ModelDataset(
                inputs=cached[f"{split}_inputs"],
                labels=cached[f"{split}_labels"],
                timestamps=cached[f"{split}_timestamps"],
                participant_ids=cached[f"{split}_participant_ids"],
            )
            for split in ("train", "val", "test")
        }


def _write_cache(path: Path, datasets: dict[str, ModelDataset]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output,
            **{
                f"{split}_{field}": getattr(dataset, field)
                for split, dataset in datasets.items()
                for field in ("inputs", "labels", "timestamps", "participant_ids")
            },
        )
    temporary.replace(path)
    for previous in path.parent.glob("*.npz"):
        if previous != path:
            previous.unlink(missing_ok=True)


def _load_generation(
    val_size: float,
    test_size: float,
    accel_path: Path,
    gt_path: Path,
    generation: str,
) -> dict[str, ModelDataset]:
    if _dataset_generation(accel_path, gt_path) != generation:
        raise RuntimeError("Prepared dataset changed while model inputs were being generated")
    connection = duckdb.connect()
    try:
        participant_rows = connection.execute(
            """
            SELECT DISTINCT participant_id
            FROM read_parquet(?)
            INTERSECT
            SELECT DISTINCT participant_id
            FROM read_parquet(?)
            ORDER BY participant_id
            """,
            [str(accel_path), str(gt_path)],
        ).fetchall()
        participant_ids = [str(row[0]) for row in participant_rows]
        if not participant_ids:
            raise RuntimeError("No participants occur in both shared Parquet files")
        if _dataset_generation(accel_path, gt_path) != generation:
            raise RuntimeError("Prepared dataset changed while model inputs were being generated")
        assignments = _load_or_create_assignments(participant_ids, val_size, test_size)
        cache_path = _cache_dir() / f"{_cache_key(generation, assignments, val_size, test_size)}.npz"
        if cache_path.is_file():
            if _dataset_generation(accel_path, gt_path) != generation:
                raise RuntimeError("Prepared dataset changed while model inputs were being generated")
            return _read_cache(cache_path)
        datasets: dict[str, list[ModelDataset]] = {"train": [], "val": [], "test": []}
        for (participant_id,) in participant_rows:
            accel, gt = _read_participant_frames(
                connection, accel_path, gt_path, str(participant_id)
            )
            dataset = _process_participant(str(participant_id), accel, gt)
            if dataset.num_examples:
                datasets[assignments[str(participant_id)]].append(dataset)
    finally:
        connection.close()
    combined = {split: _combine(parts) for split, parts in datasets.items()}
    if not combined["train"].num_examples:
        raise RuntimeError("No usable model windows were produced from the shared Parquet data")
    if _dataset_generation(accel_path, gt_path) != generation:
        raise RuntimeError("Prepared dataset changed while model inputs were being generated")
    _write_cache(cache_path, combined)
    return combined


def load_shared_datasets(val_size: float, test_size: float) -> dict[str, ModelDataset]:
    """Load a stable prepared generation, using a persistent processed cache."""
    accel_path, gt_path = _paths_from_env()
    for attempt in range(2):
        generation = _dataset_generation(accel_path, gt_path)
        try:
            return _load_generation(
                val_size, test_size, accel_path, gt_path, generation
            )
        except RuntimeError as exc:
            if "changed while" not in str(exc) or attempt:
                raise
    raise RuntimeError("Could not load a stable prepared dataset generation")


def load_shared_dataset() -> ModelDataset:
    """Backward-compatible helper that combines all local participants."""
    datasets = load_shared_datasets(val_size=0, test_size=0)
    return datasets["train"]
