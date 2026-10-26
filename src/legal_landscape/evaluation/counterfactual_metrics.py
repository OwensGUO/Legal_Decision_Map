"""Type-specific counterfactual evaluation metrics."""

from __future__ import annotations

from typing import Any

from legal_landscape.evaluation.prepared import prepare_counterfactual


def counterfactual_metrics(
    rows: list[dict[str, Any]], *, invariant_drift_threshold: float = 0.1
) -> dict[str, float | int]:
    rows = [{"group_id": row.get("group_id", "all"), **row} for row in rows]
    return prepare_counterfactual(
        rows, invariant_drift_threshold=invariant_drift_threshold
    ).metrics(None)
