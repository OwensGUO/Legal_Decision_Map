"""CMDL defendant-, case-, and defendant-count-stratified metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def cmdl_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("CMDL metrics require at least one row")
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        groups[str(row["group_id"])].append(bool(row["correct"]))
    strata: dict[str, list[float]] = defaultdict(list)
    for outcomes in groups.values():
        strata[str(len(outcomes))].append(float(all(outcomes)))
    return {
        "defendant_accuracy": float(np.mean([bool(row["correct"]) for row in rows])),
        "case_exact_accuracy": float(np.mean([all(values) for values in groups.values()])),
        "defendant_count": len(rows),
        "case_count": len(groups),
        "by_defendant_count": {
            count: {"case_count": len(values), "case_exact_accuracy": float(np.mean(values))}
            for count, values in sorted(strata.items(), key=lambda item: int(item[0]))
        },
    }
