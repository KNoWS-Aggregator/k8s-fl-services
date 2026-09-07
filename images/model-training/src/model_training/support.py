"""Training data and round-persistence helpers."""
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
