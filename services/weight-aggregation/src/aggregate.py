"""Federated averaging of participant weights, weighted by each
participant's number of training examples (standard FedAvg)."""
import numpy as np

from common.messages import TrainingMetrics
from common.weight_io import bytes_to_weights, weights_to_bytes


def federated_average(weights_by_participant: dict[str, bytes],
                       metrics_by_participant: dict[str, TrainingMetrics]) -> bytes:
    participant_ids = list(weights_by_participant.keys())
    decoded = {pid: bytes_to_weights(weights_by_participant[pid]) for pid in participant_ids}
    num_examples = {pid: metrics_by_participant[pid].num_examples for pid in participant_ids}
    total_examples = sum(num_examples.values())

    num_layers = len(decoded[participant_ids[0]])
    averaged: list[np.ndarray] = []

    for layer_idx in range(num_layers):
        weighted_sum = sum(
            decoded[pid][layer_idx] * (num_examples[pid] / total_examples)
            for pid in participant_ids
        )
        averaged.append(weighted_sum)

    return weights_to_bytes(averaged)