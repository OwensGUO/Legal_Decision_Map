"""Static multi-label charge and sentence metrics."""

from __future__ import annotations

import numpy as np


def _f1(tp: float, fp: float, fn: float) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def charge_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict[str, float]:
    truth = np.asarray(true_labels, dtype=bool)
    prediction = np.asarray(predicted_labels, dtype=bool)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("charge labels must be equally shaped two-dimensional arrays")
    tp = np.logical_and(truth, prediction).sum(axis=0)
    fp = np.logical_and(~truth, prediction).sum(axis=0)
    fn = np.logical_and(truth, ~prediction).sum(axis=0)
    macro = float(
        np.mean([_f1(float(a), float(b), float(c)) for a, b, c in zip(tp, fp, fn, strict=True)])
    )
    return {
        "macro_f1": macro,
        "micro_f1": _f1(float(tp.sum()), float(fp.sum()), float(fn.sum())),
        "exact_match": float(np.mean(np.all(truth == prediction, axis=1))),
    }


def sentence_metrics(
    predicted_months: np.ndarray,
    true_months: np.ndarray,
    penalty_types: tuple[str, ...] | list[str],
    *,
    charge_counts: np.ndarray | None = None,
    tolerance_months: float = 3.0,
) -> dict[str, float | int]:
    """Month errors on fixed-term cases; with ``charge_counts``, single-charge cases only.

    Training regresses months only for single-charge fixed-term cases, so evaluation passes
    ``charge_counts`` to use the same eligibility.
    """
    predicted = np.asarray(predicted_months, dtype=float)
    truth = np.asarray(true_months, dtype=float)
    penalties = np.asarray(penalty_types)
    if not (predicted.shape == truth.shape == penalties.shape):
        raise ValueError("sentence arrays must have equal shape")
    mask = (penalties == "fixed_term") & np.isfinite(predicted) & np.isfinite(truth)
    if charge_counts is not None:
        counts = np.asarray(charge_counts)
        if counts.shape != predicted.shape:
            raise ValueError("charge_counts must match the sentence arrays")
        mask &= counts == 1
    if not mask.any():
        return {
            "eligible_count": 0,
            "mae": float("nan"),
            "log_mae": float("nan"),
            "tolerance_accuracy": float("nan"),
        }
    errors = np.abs(predicted[mask] - truth[mask])
    log_errors = np.abs(
        np.log1p(np.maximum(predicted[mask], 0)) - np.log1p(np.maximum(truth[mask], 0))
    )
    return {
        "eligible_count": int(mask.sum()),
        "mae": float(errors.mean()),
        "log_mae": float(log_errors.mean()),
        "tolerance_accuracy": float(np.mean(errors <= tolerance_months)),
    }
