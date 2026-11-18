"""Five fixed columns, raw-score argmax, explicit zero-denominator metrics."""
from __future__ import annotations

import numpy as np

from .config import CLASSES


def classification_metrics(truth: np.ndarray, scores: np.ndarray) -> dict:
    truth, scores = np.asarray(truth), np.asarray(scores, dtype=np.float64)
    if truth.ndim != 1 or not len(truth) or truth.dtype.kind not in "iu" or np.any(truth < 0) or np.any(truth > 4):
        raise ValueError("true labels must be integer class IDs 0..4")
    if scores.shape != (len(truth), 5) or not np.isfinite(scores).all():
        raise ValueError("scores must be finite with five fixed columns")
    prediction = scores.argmax(axis=1)
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (truth, prediction), 1)
    true_count, pred_count = confusion.sum(axis=1), confusion.sum(axis=0)
    tp = confusion.diagonal()
    recall = np.divide(tp, true_count, out=np.zeros(5), where=true_count != 0)
    f1 = np.divide(2 * tp, true_count + pred_count, out=np.zeros(5), where=(true_count + pred_count) != 0)
    return {"classes": list(CLASSES), "n_evaluated": int(len(truth)),
            "accuracy": float(tp.sum() / len(truth)), "macro_f1": float(f1.mean()),
            "per_class_recall": recall.tolist(), "confusion_matrix": confusion.tolist()}
