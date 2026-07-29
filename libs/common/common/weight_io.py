"""Serialize/deserialize model weights (lists of ndarrays) to/from raw
bytes, for embedding in inter-service messages"""
import io

import numpy as np


def weights_to_bytes(weights: list[np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **{f"arr_{i}": w for i, w in enumerate(weights)})
    return buffer.getvalue()


def bytes_to_weights(data: bytes) -> list[np.ndarray]:
    buffer = io.BytesIO(data)
    with np.load(buffer, allow_pickle=True) as npz:
        keys = sorted(npz.files, key=lambda k: int(k.split("_")[-1]))
        return [npz[k] for k in keys]