"""Group-clustered bootstrap and Holm multiple-comparison correction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np


def _group_rows(
    rows: list[dict[str, Any]], group_key: str
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    return dict(groups)


def _interval(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan"), float("nan")
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    *,
    iterations: int = 2000,
    seed: int = 42,
    group_key: str = "group_id",
) -> dict[str, float | int]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    groups = _group_rows(rows, group_key)
    if not groups:
        raise ValueError("bootstrap requires at least one group")
    keys = tuple(groups)
    generator = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        selected = generator.choice(keys, size=len(keys), replace=True)
        sample = [row for key in selected for row in groups[str(key)]]
        estimates[index] = metric(sample)
    lower, upper = _interval(estimates)
    return {
        "estimate": float(metric(rows)),
        "lower": lower,
        "upper": upper,
        "iterations": iterations,
        "groups": len(groups),
    }


def bootstrap_metric_set(
    rows: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], dict[str, float | int]],
    names: tuple[str, ...],
    *,
    iterations: int = 2000,
    seed: int = 42,
    group_key: str = "group_id",
) -> dict[str, dict[str, float | int]]:
    """Bootstrap several scalar metrics using the same sampled group clusters."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not names:
        raise ValueError("at least one metric name is required")
    groups = _group_rows(rows, group_key)
    if not groups:
        raise ValueError("bootstrap requires at least one group")
    estimates = metric(rows)
    missing = [name for name in names if name not in estimates]
    if missing:
        raise KeyError(f"metric function did not return: {', '.join(missing)}")
    samples = {name: np.empty(iterations, dtype=float) for name in names}
    keys = tuple(groups)
    generator = np.random.default_rng(seed)
    for index in range(iterations):
        selected = generator.choice(keys, size=len(keys), replace=True)
        sample = [row for key in selected for row in groups[str(key)]]
        values = metric(sample)
        for name in names:
            samples[name][index] = float(values[name])
    result: dict[str, dict[str, float | int]] = {}
    for name in names:
        lower, upper = _interval(samples[name])
        result[name] = {
            "estimate": float(estimates[name]),
            "lower": lower,
            "upper": upper,
            "iterations": iterations,
            "groups": len(groups),
        }
    return result


def paired_cluster_test(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    *,
    iterations: int = 2000,
    seed: int = 42,
    group_key: str = "group_id",
) -> dict[str, float | int]:
    """Compare two systems with paired whole-group bootstrap resampling."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    candidate_groups = _group_rows(candidate, group_key)
    reference_groups = _group_rows(reference, group_key)
    if not candidate_groups or set(candidate_groups) != set(reference_groups):
        raise ValueError("paired comparison requires identical group IDs")
    keys = tuple(sorted(candidate_groups))
    generator = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=float)
    for index in range(iterations):
        selected = generator.choice(keys, size=len(keys), replace=True)
        candidate_sample = [
            row for key in selected for row in candidate_groups[str(key)]
        ]
        reference_sample = [
            row for key in selected for row in reference_groups[str(key)]
        ]
        differences[index] = float(metric(candidate_sample)) - float(metric(reference_sample))
    finite = differences[np.isfinite(differences)]
    if not finite.size:
        p_value = float("nan")
    else:
        lower_tail = (float(np.count_nonzero(finite <= 0.0)) + 1.0) / (len(finite) + 1.0)
        upper_tail = (float(np.count_nonzero(finite >= 0.0)) + 1.0) / (len(finite) + 1.0)
        p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    lower, upper = _interval(differences)
    return {
        "difference": float(metric(candidate)) - float(metric(reference)),
        "lower": lower,
        "upper": upper,
        "p_value": p_value,
        "iterations": iterations,
        "groups": len(keys),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values keyed like the input."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    previous = 0.0
    adjusted: dict[str, float] = {}
    for index, (name, value) in enumerate(ordered):
        current = min(1.0, (total - index) * float(value))
        previous = max(previous, current)
        adjusted[name] = previous
    return adjusted
