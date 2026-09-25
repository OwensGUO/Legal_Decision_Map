"""Static classification metrics plus eligible-case sentence regression errors."""

from __future__ import annotations

import math

import numpy as np

SENTENCE_CLASS_VOCABULARY = (
    "fixed_term_0_6",
    "fixed_term_7_12",
    "fixed_term_13_24",
    "fixed_term_25_36",
    "fixed_term_37_60",
    "fixed_term_61_120",
    "fixed_term_121_plus",
    "life",
    "death",
    "detention",
    "control",
    "exempt",
    "unknown",
)


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _f1(tp: float, fp: float, fn: float) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def multilabel_metrics(
    true_labels: np.ndarray, predicted_labels: np.ndarray
) -> dict[str, float]:
    """Return exact-set accuracy plus macro/micro multi-label metrics."""
    truth = np.asarray(true_labels, dtype=bool)
    prediction = np.asarray(predicted_labels, dtype=bool)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("labels must be equally shaped two-dimensional arrays")
    if truth.shape[0] == 0 or truth.shape[1] == 0:
        raise ValueError("multi-label arrays must not be empty")
    tp = np.logical_and(truth, prediction).sum(axis=0)
    fp = np.logical_and(~truth, prediction).sum(axis=0)
    fn = np.logical_and(truth, ~prediction).sum(axis=0)
    macro_precision = float(
        np.mean([_ratio(float(a), float(a + b)) for a, b in zip(tp, fp, strict=True)])
    )
    macro_recall = float(
        np.mean([_ratio(float(a), float(a + c)) for a, c in zip(tp, fn, strict=True)])
    )
    macro_f1 = float(
        np.mean(
            [_f1(float(a), float(b), float(c)) for a, b, c in zip(tp, fp, fn, strict=True)]
        )
    )
    total_tp = float(tp.sum())
    total_fp = float(fp.sum())
    total_fn = float(fn.sum())
    return {
        "accuracy": float(np.mean(np.all(truth == prediction, axis=1))),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": _ratio(total_tp, total_tp + total_fp),
        "micro_recall": _ratio(total_tp, total_tp + total_fn),
        "micro_f1": _f1(total_tp, total_fp, total_fn),
        "hamming_accuracy": float(np.mean(truth == prediction)),
    }


def charge_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict[str, float]:
    """Charge metrics with aliases retained for older evaluation outputs."""
    result = multilabel_metrics(true_labels, predicted_labels)
    return {**result, "exact_match": result["accuracy"]}


def multiclass_metrics(
    true_labels: tuple[str, ...] | list[str] | np.ndarray,
    predicted_labels: tuple[str, ...] | list[str] | np.ndarray,
    *,
    labels: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    """Return accuracy and macro/micro metrics for single-label classification."""
    truth = np.asarray(true_labels, dtype=str)
    prediction = np.asarray(predicted_labels, dtype=str)
    if truth.shape != prediction.shape or truth.ndim != 1:
        raise ValueError("labels must be equally shaped one-dimensional arrays")
    if truth.size == 0:
        raise ValueError("multi-class arrays must not be empty")
    vocabulary = tuple(labels or sorted(set(truth.tolist()) | set(prediction.tolist())))
    if not vocabulary:
        raise ValueError("multi-class vocabulary must not be empty")
    unknown = (set(truth.tolist()) | set(prediction.tolist())) - set(vocabulary)
    if unknown:
        raise ValueError(f"labels outside the declared vocabulary: {sorted(unknown)}")
    per_class: list[tuple[float, float, float]] = []
    for label in vocabulary:
        tp = float(np.logical_and(truth == label, prediction == label).sum())
        fp = float(np.logical_and(truth != label, prediction == label).sum())
        fn = float(np.logical_and(truth == label, prediction != label).sum())
        per_class.append((tp, fp, fn))
    return {
        "accuracy": float(np.mean(truth == prediction)),
        "macro_precision": float(
            np.mean([_ratio(tp, tp + fp) for tp, fp, _fn in per_class])
        ),
        "macro_recall": float(np.mean([_ratio(tp, tp + fn) for tp, _fp, fn in per_class])),
        "macro_f1": float(np.mean([_f1(tp, fp, fn) for tp, fp, fn in per_class])),
        # For single-label classification, all three micro metrics equal accuracy.
        "micro_precision": float(np.mean(truth == prediction)),
        "micro_recall": float(np.mean(truth == prediction)),
        "micro_f1": float(np.mean(truth == prediction)),
    }


def sentence_class(penalty_type: str, months: float | int | None) -> str:
    """Map a mixed penalty/month prediction to a deterministic evaluation class."""
    if penalty_type != "fixed_term":
        return penalty_type if penalty_type in SENTENCE_CLASS_VOCABULARY else "unknown"
    if months is None or not math.isfinite(float(months)) or float(months) < 0:
        return "unknown"
    value = float(months)
    for upper, label in (
        (6, "fixed_term_0_6"),
        (12, "fixed_term_7_12"),
        (24, "fixed_term_13_24"),
        (36, "fixed_term_25_36"),
        (60, "fixed_term_37_60"),
        (120, "fixed_term_61_120"),
    ):
        if value <= upper:
            return label
    return "fixed_term_121_plus"


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
