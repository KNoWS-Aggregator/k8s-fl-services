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
) -> dict:
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

    result = {
        "num_examples": total_examples,
        "eval_loss": weighted("eval_loss"),
        "eval_accuracy": weighted("eval_accuracy"),
        "eval_f1_macro": weighted("eval_f1_macro"),
        "eval_f1_weighted": weighted("eval_f1_weighted"),
        "eval_precision_macro": weighted("eval_precision_macro"),
        "eval_recall_macro": weighted("eval_recall_macro"),
        "confusion_matrix": None,
        "per_class": None,
    }

    # A summed confusion matrix produces exact global classification metrics.
    # Fall back to example-weighted scalar metrics while older clients are
    # being upgraded and do not yet provide a matrix.
    matrices = [metrics.confusion_matrix for metrics in metrics_by_client.values()]
    if not matrices or any(matrix is None for matrix in matrices):
        return result
    arrays = [np.asarray(matrix, dtype=np.int64) for matrix in matrices]
    shape = arrays[0].shape
    if len(shape) != 2 or shape[0] != shape[1] or any(array.shape != shape for array in arrays):
        raise ValueError("Client confusion matrices must be square and have matching dimensions")
    matrix = np.sum(arrays, axis=0)
    if np.any(matrix < 0):
        raise ValueError("Client confusion matrices cannot contain negative counts")
    if int(matrix.sum()) != total_examples:
        raise ValueError("Confusion-matrix counts do not match evaluation num_examples")

    true_support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    true_positives = np.diag(matrix)
    precision = np.divide(
        true_positives, predicted_support,
        out=np.zeros(shape[0], dtype=float), where=predicted_support != 0,
    )
    recall = np.divide(
        true_positives, true_support,
        out=np.zeros(shape[0], dtype=float), where=true_support != 0,
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(shape[0], dtype=float), where=(precision + recall) != 0,
    )
    class_names = next(
        (list(metrics.per_class) for metrics in metrics_by_client.values() if metrics.per_class),
        [str(index) for index in range(shape[0])],
    )
    if len(class_names) != shape[0]:
        raise ValueError("Per-class labels do not match confusion-matrix dimensions")
    if any(
        metrics.per_class and list(metrics.per_class) != class_names
        for metrics in metrics_by_client.values()
    ):
        raise ValueError("Clients must use the same class order")

    result.update(
        eval_accuracy=float(true_positives.sum() / total_examples),
        eval_f1_macro=float(f1.mean()),
        eval_f1_weighted=float(np.average(f1, weights=true_support)),
        eval_precision_macro=float(precision.mean()),
        eval_recall_macro=float(recall.mean()),
        confusion_matrix=matrix.tolist(),
        per_class={
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(true_support[index]),
            }
            for index, name in enumerate(class_names)
        },
    )
    return result
