"""Calibration metrics for binary probability predictions (ECE, reliability bins)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def reliability_bins(
    prob: np.ndarray,
    y: np.ndarray,
    n_bins: int = 15,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-width bins on predicted probability.

    Returns (mean_pred, frac_pos, count) per bin (only bins with count>0 may be used for plots).
    """
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred: List[float] = []
    frac_pos: List[float] = []
    counts: List[int] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            mask = (prob >= lo) & (prob <= hi + eps)
        else:
            mask = (prob >= lo) & (prob < hi)
        cnt = int(mask.sum())
        counts.append(cnt)
        if cnt == 0:
            mean_pred.append(float("nan"))
            frac_pos.append(float("nan"))
        else:
            mean_pred.append(float(prob[mask].mean()))
            frac_pos.append(float(y[mask].mean()))
    return np.array(mean_pred), np.array(frac_pos), np.array(counts, dtype=np.int64)


def expected_calibration_error(
    prob: np.ndarray,
    y: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Standard ECE: sum over bins of (|acc - conf| * weight)."""
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(prob) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = float(len(prob))
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            mask = (prob >= lo) & (prob <= 1.0)
        else:
            mask = (prob >= lo) & (prob < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        conf = float(prob[mask].mean())
        acc = float(y[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def calibration_report(prob: np.ndarray, y: np.ndarray, n_bins: int = 15) -> Dict[str, object]:
    mp, fp, ct = reliability_bins(prob, y, n_bins=n_bins)
    return {
        "ece": expected_calibration_error(prob, y, n_bins=n_bins),
        "n_bins": n_bins,
        "n_samples": int(len(np.asarray(prob).reshape(-1))),
        "reliability": {
            "mean_pred": np.nan_to_num(mp, nan=-1.0).tolist(),
            "frac_positive": np.nan_to_num(fp, nan=-1.0).tolist(),
            "counts": ct.tolist(),
        },
    }
