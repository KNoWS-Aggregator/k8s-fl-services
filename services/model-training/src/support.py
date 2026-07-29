"""Run-naming and result-persistence helpers"""
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_class_weight

from common.messages import TrainingConfig

def load_data(d_set: str, scaling: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load this instance's local dataset from a .npz file, located via
    the DATASET_PATH environment variable.

    Args:
        d_set: which split to load ("train", "val", "test").
        scaling: whether to apply per-channel MinMax scaling.

    Returns:
        (X_data, y_data, timestamps)
    """
    dataset_path = Path(os.environ["DATASET_PATH"])

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.load(dataset_path)
    data_set = d_set.lower().strip()
    assert data_set in ("train", "val", "test"), "d_set must be 'train', 'val', or 'test'"

    x_data = data[f"{data_set}_input"]
    if scaling and x_data.shape[0] > 0:
        scaled_channels = []
        for i in range(x_data.shape[2]):
            scaler = MinMaxScaler()
            if data_set == "train":
                scaled_channels.append(scaler.fit_transform(data["train_input"][:, :, i]))
            else:
                scaler.fit(data["train_input"][:, :, i])
                scaled_channels.append(scaler.transform(x_data[:, :, i]))
        x_data = np.stack(scaled_channels, axis=2)

    return x_data, data[f"{data_set}_label"], data[f"{data_set}_timestamps"]


def get_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights for one-hot encoded labels."""
    sample_label_number = np.argmax(labels, axis=1)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(sample_label_number),
        y=sample_label_number,
    )
    return {k: weights[i] for i, k in enumerate(np.unique(sample_label_number))}

def get_save_name(config: TrainingConfig) -> str:
    """Build a descriptive folder name for one training run, derived from
    its hyperparameters, so re-runs with different settings land in
    different folders instead of overwriting each other.
    """
    val_size = config.val_size * 100 if config.val_size < 1 else config.val_size
    test_size = config.test_size * 100 if config.test_size < 1 else config.test_size

    base = (
        f"FL_{config.model_prefix}"
        f"_rounds_{config.num_rounds}"
        f"_epochs_{config.local_epochs}"
        f"_batch_{config.batch_size}"
        f"_scaling_{config.scaling}"
        f"_balance_{config.balance}"
    )

    if config.group_size > 0:
        return f"{base}_groups_{config.group_size}_validation_{val_size:.0f}_test_{test_size:.0f}"
    return f"{base}_validation_{val_size:.0f}_test_{test_size:.0f}_gap_{config.gap:.0f}"


def save_training_history(n_round: int, client_name: str, history: dict, base_path: Path) -> None:
    """Save one client's per-round training history (loss/accuracy curves)
    as JSON under base_path/round_{n_round}/{client_name}_history.json.
    """
    save_path = base_path / f"round_{n_round}" / f"{client_name}_history.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f)