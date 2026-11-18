"""Coarse ordered pooling of already normalized frozen frame embeddings."""
from __future__ import annotations

import numpy as np
from .clip_features import pool_normalized

SEGMENTS = ((0, 6), (6, 11), (11, 16))


def temporal_pool(z: np.ndarray, representation: str) -> np.ndarray:
    z = np.asarray(z)
    if (z.ndim != 2 or z.shape[0] != 16 or z.shape[1] < 1 or not np.isfinite(z).all()
            or not np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)):
        raise ValueError('expected finite, unit-normalized 16 x d frame features')
    if representation == 'V0':
        return pool_normalized(z)  # Preserve first-round arithmetic exactly.
    if representation != 'V1':
        raise ValueError('representation must be V0 or V1')
    statistics = z.astype(np.float64)
    blocks = [statistics.mean(axis=0), statistics.std(axis=0, ddof=0),
              *[statistics[start:end].mean(axis=0) for start, end in SEGMENTS]]
    vector = np.concatenate([block / (np.linalg.norm(block) + 1e-6) for block in blocks])
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-6:
        raise ValueError('zero or nonfinite temporal feature')
    return (vector / norm).astype(np.float32)
