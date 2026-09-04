"""Run-naming and result-persistence helpers"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_class_weight

from common.messages import TrainingConfig
from .data_pipeline import ModelDataset, load_shared_datasets


def round_dir(session_id: str, client_id: str, round_id: int) -> Path:
    return (
        Path(os.getenv("DATA_DIR", "/app/data"))
        / "model-training"
        / "rounds"
        / session_id
        / client_id
        / f"round_{round_id}"
    )


def load_data(config: TrainingConfig) -> dict[str, ModelDataset]:
    """Load participant-level splits and optionally scale from training only."""
    datasets = load_shared_datasets(config.val_size, config.test_size)
    if not config.scaling:
        return datasets
    train = datasets["train"]
    scaled: dict[str, ModelDataset] = {}
    scalers = []
    for channel in range(train.inputs.shape[2]):
        scaler = MinMaxScaler()
        scaler.fit(train.inputs[:, :, channel])
        scalers.append(scaler)
    for split, dataset in datasets.items():
        inputs = dataset.inputs.copy()
        if dataset.num_examples:
            inputs = np.stack(
                [
                    scalers[channel].transform(inputs[:, :, channel])
                    for channel in range(inputs.shape[2])
                ],
                axis=2,
            ).astype(np.float32)
        scaled[split] = ModelDataset(
            inputs=inputs,
            labels=dataset.labels,
            timestamps=dataset.timestamps,
            participant_ids=dataset.participant_ids,
        )
    return scaled


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
