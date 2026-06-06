"""Evaluation utilities: F2, threshold sweep, PR metrics, fairness slice."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)


def f2(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    return float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))


def best_threshold_for_f2(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, float]:
    """Sweep thresholds 0.01–0.99 and return (best_threshold, best_f2)."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    scores = [f2(y_true, y_prob, float(t)) for t in thresholds]
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])


def evaluate_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold": threshold,
    }


def fairness_report(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    segment_col: pd.Series,
    segment_name: str = "segment",
    min_samples: int = 100,
) -> pd.DataFrame:
    """Compute FPR and recall per segment value.

    PaySim lacks demographic columns; slice on 'type' as a proxy.
    Note: this is structural fairness (FPR parity), not demographic fairness.
    """
    results = []
    for segment in sorted(segment_col.unique()):
        mask = segment_col == segment
        if mask.sum() < min_samples:
            continue
        yt = y_true[mask].values
        yp = y_prob[mask]
        metrics = evaluate_at_threshold(yt, yp, threshold)
        metrics[segment_name] = segment
        metrics["n"] = int(mask.sum())
        results.append(metrics)
    df = pd.DataFrame(results)
    if not df.empty:
        cols = [segment_name, "n", "fpr", "recall", "precision", "f2"]
        df = df[[c for c in cols if c in df.columns]]
    return df
