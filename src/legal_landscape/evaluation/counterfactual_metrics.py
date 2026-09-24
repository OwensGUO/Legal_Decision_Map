"""Type-specific counterfactual evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def counterfactual_metrics(
    rows: list[dict[str, Any]], *, invariant_drift_threshold: float = 0.1
) -> dict[str, float | int]:
    flip_correct: list[float] = []
    invariant_violations: list[float] = []
    invariant_drifts: list[float] = []
    rank_correct: list[float] = []
    for row in rows:
        kind = row.get("intervention_type")
        if kind == "charge_flip":
            probabilities = np.asarray(row["counterfactual_charge_probabilities"], dtype=float)
            flip_correct.append(float(int(probabilities.argmax()) == int(row["target_charge"])))
        elif kind == "invariant":
            parent = np.asarray(row["parent_charge_probabilities"], dtype=float)
            counterfactual = np.asarray(row["counterfactual_charge_probabilities"], dtype=float)
            drift = float(np.max(np.abs(parent - counterfactual)))
            invariant_drifts.append(drift)
            invariant_violations.append(
                float(
                    parent.argmax() != counterfactual.argmax() or drift > invariant_drift_threshold
                )
            )
        elif kind == "sentence_rank":
            change = float(row["counterfactual_sentence"]) - float(row["parent_sentence"])
            direction = int(row["rank_direction"])
            rank_correct.append(float(direction * change > 0))
    pairwise = _mean(rank_correct)
    return {
        "flip_count": len(flip_correct),
        "flip_accuracy": _mean(flip_correct),
        "invariant_count": len(invariant_violations),
        "invariant_violation_rate": _mean(invariant_violations),
        "sentence_rank_count": len(rank_correct),
        "sentence_pairwise_accuracy": pairwise,
        "monotonicity_violation_rate": 1.0 - pairwise if rank_correct else float("nan"),
        "non_target_prediction_drift": _mean(invariant_drifts),
    }
