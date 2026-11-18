"""Centered dual solution to mean-sample multi-output L2 ridge."""
from __future__ import annotations

import numpy as np


def fit_ridge(x: np.ndarray, labels: np.ndarray, lambda_: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    x, labels = np.asarray(x, dtype=np.float64), np.asarray(labels)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("ridge requires finite n x d features")
    if labels.shape != (len(x),) or labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= 5):
        raise ValueError("ridge labels must be integer class IDs 0..4")
    if not np.isfinite(lambda_) or lambda_ <= 0:
        raise ValueError("lambda must be positive and finite")
    targets = np.eye(5, dtype=np.float64)[labels]
    mean_x, mean_y = x.mean(axis=0), targets.mean(axis=0)
    centered_x, centered_y = x - mean_x, targets - mean_y
    dual = np.linalg.solve(centered_x @ centered_x.T + len(x) * lambda_ * np.eye(len(x)), centered_y)
    weights = centered_x.T @ dual
    intercept = mean_y - mean_x @ weights
    return weights, intercept
