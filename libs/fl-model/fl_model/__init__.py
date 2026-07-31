"""Shared federated model definition."""

import hashlib
import json

from .model import ARCHITECTURE, HYPER_PARAMETERS, RotationLayer, load_model

MODEL_VERSION = 1


def weight_signature(weights) -> str:
    """Return a stable signature for an ordered model-weight structure."""
    structure = {
        "model_version": MODEL_VERSION,
        "architecture": ARCHITECTURE,
        "hyper_parameters": HYPER_PARAMETERS,
        "weights": [
            {"shape": list(weight.shape), "dtype": str(weight.dtype)}
            for weight in weights
        ],
    }
    digest = hashlib.sha256(
        json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"sha256:{digest}"


def create_initial_weights():
    """Create one canonical freshly initialized model weight set."""
    model = load_model()
    weights = model.get_weights()
    return weights, weight_signature(weights)

__all__ = [
    "ARCHITECTURE",
    "HYPER_PARAMETERS",
    "MODEL_VERSION",
    "RotationLayer",
    "create_initial_weights",
    "load_model",
    "weight_signature",
]
