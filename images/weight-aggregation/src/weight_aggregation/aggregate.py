"""Federated averaging of client weights, weighted by each
participant's number of training examples (standard FedAvg)."""
import numpy as np

from common.messages import EvaluationMetrics, TrainingMetrics
from common.weight_io import bytes_to_weights, weights_to_bytes


def federated_average(weights_by_client: dict[str, bytes],
                      metrics_by_client: dict[str, TrainingMetrics]) -> bytes:
    client_ids = list(weights_by_client.keys())
    decoded = {client_id: bytes_to_weights(weights_by_client[client_id]) for client_id in client_ids}
    num_examples = {client_id: metrics_by_client[client_id].num_examples for client_id in client_ids}
    total_examples = sum(num_examples.values())

    num_layers = len(decoded[client_ids[0]])
    averaged: list[np.ndarray] = []

    for layer_idx in range(num_layers):
        weighted_sum = sum(
            decoded[client_id][layer_idx] * (num_examples[client_id] / total_examples)
            for client_id in client_ids
        )
        averaged.append(weighted_sum)

    return weights_to_bytes(averaged)


def aggregate_evaluation_metrics(
    metrics_by_client: dict[str, EvaluationMetrics],
) -> dict[str, float | int | None]:
    """Example-weighted aggregation of client evaluation metrics."""
    if not metrics_by_client:
        raise ValueError("No evaluation metrics were supplied")
    total_examples = sum(metrics.num_examples for metrics in metrics_by_client.values())

    def weighted(field: str) -> float | None:
        available = [
            metrics
            for metrics in metrics_by_client.values()
            if getattr(metrics, field) is not None
        ]
        examples = sum(metrics.num_examples for metrics in available)
        if not examples:
            return None
        return sum(
            float(getattr(metrics, field)) * metrics.num_examples
            for metrics in available
        ) / examples

    return {
        "num_examples": total_examples,
        "eval_loss": weighted("eval_loss"),
        "eval_accuracy": weighted("eval_accuracy"),
    }
