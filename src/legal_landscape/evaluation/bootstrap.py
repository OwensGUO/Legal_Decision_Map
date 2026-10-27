"""Group-clustered bootstrap and Holm multiple-comparison correction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from legal_landscape.progress import NullProgressReporter, ProgressReporter


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
    progress: ProgressReporter | None = None,
) -> dict[str, float | int]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    groups = _group_rows(rows, group_key)
    if not groups:
        raise ValueError("bootstrap requires at least one group")
    keys = tuple(groups)
    generator = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    reporter = progress or NullProgressReporter()
    with reporter.task(
        "Bootstrap confidence intervals", total=iterations, groups=len(groups)
    ) as task:
        for index in range(iterations):
            selected = generator.choice(keys, size=len(keys), replace=True)
            sample = [row for key in selected for row in groups[str(key)]]
            estimates[index] = metric(sample)
            task.advance()
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
    progress: ProgressReporter | None = None,
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
    reporter = progress or NullProgressReporter()
    with reporter.task(
        "Bootstrap confidence intervals", total=iterations, groups=len(groups)
    ) as task:
        for index in range(iterations):
            selected = generator.choice(keys, size=len(keys), replace=True)
            sample = [row for key in selected for row in groups[str(key)]]
            values = metric(sample)
            for name in names:
                samples[name][index] = float(values[name])
            task.advance()
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


def _sample_key(row: dict[str, Any]) -> tuple[Any, ...]:
    if row.get("intervention_type"):
        generation_id = row.get("generation_id")
        if not isinstance(generation_id, str) or not generation_id:
            raise ValueError(
                "counterfactual paired row lacks generation_id; "
                "the parent case_id is not a stable pair key"
            )
        return ("generation", generation_id)
    case_id = row.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("paired comparison row lacks case_id or generation_id")
    return ("case", case_id)


def _label_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    return tuple(value)


def _truth_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    if "charge_vocabulary" not in row:
        raise ValueError("paired comparison row lacks charge_vocabulary")
    article_vocabulary = row.get("article_vocabulary")
    return (
        tuple(row["charge_vocabulary"]),
        None if article_vocabulary is None else tuple(article_vocabulary),
        _label_tuple(row.get("true_charges")),
        _label_tuple(row.get("unseen_true_charge_labels")),
        None if "true_articles" not in row else _label_tuple(row.get("true_articles")),
        _label_tuple(row.get("unseen_true_article_labels")),
        row.get("articles_available", "missing"),
        row.get("penalty_type"),
        row.get("true_months"),
        row.get("intervention_type"),
        row.get("rank_direction"),
        row.get("target_charge"),
        row.get("rule_id"),
    )


def assert_paired_rows(
    candidate: list[dict[str, Any]], reference: list[dict[str, Any]]
) -> None:
    """Require the same samples, groups, truths, vocabularies, and interventions.

    Row order may differ. Missing identity, repeated rows, or disagreeing truths are errors.
    Counterfactual rows pair on generation_id, not on the parent case alone.
    """
    def index(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
        found: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in rows:
            key = _sample_key(row)
            if key in found:
                raise ValueError(f"paired comparison has a repeated sample {key}")
            found[key] = row
        return found

    left, right = index(candidate), index(reference)
    if set(left) != set(right):
        missing = sorted(set(left) ^ set(right))[:3]
        raise ValueError(f"paired comparison sample sets differ, for example {missing}")
    for key, row in left.items():
        other = right[key]
        if str(row.get("group_id")) != str(other.get("group_id")):
            raise ValueError(f"sample {key} is in different groups")
        if _truth_signature(row) != _truth_signature(other):
            raise ValueError(f"sample {key} has different truth or intervention fields")


def paired_cluster_test(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    *,
    iterations: int = 2000,
    seed: int = 42,
    group_key: str = "group_id",
    progress: ProgressReporter | None = None,
    description: str = "Paired bootstrap",
) -> dict[str, float | int]:
    """Compare two systems with paired whole-group bootstrap resampling."""
    assert_paired_rows(candidate, reference)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    candidate_groups = _group_rows(candidate, group_key)
    reference_groups = _group_rows(reference, group_key)
    if not candidate_groups or set(candidate_groups) != set(reference_groups):
        raise ValueError("paired comparison requires identical group IDs")
    keys = tuple(sorted(candidate_groups))
    generator = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=float)
    reporter = progress or NullProgressReporter()
    with reporter.task(description, total=iterations, groups=len(keys)) as task:
        for index in range(iterations):
            selected = generator.choice(keys, size=len(keys), replace=True)
            candidate_sample = [
                row for key in selected for row in candidate_groups[str(key)]
            ]
            reference_sample = [
                row for key in selected for row in reference_groups[str(key)]
            ]
            differences[index] = float(metric(candidate_sample)) - float(metric(reference_sample))
            task.advance()
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


def _group_count_replicates(groups: int, iterations: int, seed: int):
    """Per-group multiplicities; draws match ``generator.choice(keys, ...)`` over the same keys."""
    generator = np.random.default_rng(seed)
    for _ in range(iterations):
        selected = generator.choice(groups, size=groups, replace=True)
        yield np.bincount(selected, minlength=groups).astype(float)


def fast_bootstrap_metric_set(
    prepared: Any,
    names: tuple[str, ...],
    *,
    iterations: int = 2000,
    seed: int = 42,
    progress: ProgressReporter | None = None,
) -> dict[str, dict[str, float | int]]:
    """``bootstrap_metric_set`` over a prepared metric object, resampling by group weights."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not names:
        raise ValueError("at least one metric name is required")
    estimates = prepared.metrics(None)
    missing = [name for name in names if name not in estimates]
    if missing:
        raise KeyError(f"metric function did not return: {', '.join(missing)}")
    groups = len(prepared.group_keys)
    samples = {name: np.empty(iterations, dtype=float) for name in names}
    reporter = progress or NullProgressReporter()
    with reporter.task("Bootstrap confidence intervals", total=iterations, groups=groups) as task:
        for index, counts in enumerate(_group_count_replicates(groups, iterations, seed)):
            values = prepared.metrics(counts)
            for name in names:
                samples[name][index] = float(values[name])
            task.advance()
    result: dict[str, dict[str, float | int]] = {}
    for name in names:
        lower, upper = _interval(samples[name])
        result[name] = {
            "estimate": float(estimates[name]),
            "lower": lower,
            "upper": upper,
            "iterations": iterations,
            "groups": groups,
        }
    return result


def fast_paired_test(
    candidate: Any,
    reference: Any,
    names: tuple[str, ...],
    *,
    iterations: int = 2000,
    seed: int = 42,
    progress: ProgressReporter | None = None,
    description: str = "Paired bootstrap",
) -> dict[str, dict[str, float | int]]:
    """``paired_cluster_test`` for several endpoints from one shared resampling sequence."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    candidate_rows = getattr(candidate, "source_rows", None)
    reference_rows = getattr(reference, "source_rows", None)
    if not isinstance(candidate_rows, list) or not isinstance(reference_rows, list):
        raise ValueError(
            "paired comparison requires source row identity on both prepared inputs"
        )
    assert_paired_rows(candidate_rows, reference_rows)
    if set(candidate.group_keys) != set(reference.group_keys):
        raise ValueError("paired comparison requires identical group IDs")
    keys = tuple(sorted(candidate.group_keys))
    position = {key: index for index, key in enumerate(keys)}
    candidate_order = np.asarray([position[key] for key in candidate.group_keys])
    reference_order = np.asarray([position[key] for key in reference.group_keys])
    differences = {name: np.empty(iterations, dtype=float) for name in names}
    reporter = progress or NullProgressReporter()
    with reporter.task(description, total=iterations, groups=len(keys)) as task:
        for index, counts in enumerate(_group_count_replicates(len(keys), iterations, seed)):
            first = candidate.metrics(counts[candidate_order])
            second = reference.metrics(counts[reference_order])
            for name in names:
                differences[name][index] = float(first[name]) - float(second[name])
            task.advance()
    full_candidate, full_reference = candidate.metrics(None), reference.metrics(None)
    result: dict[str, dict[str, float | int]] = {}
    for name in names:
        values = differences[name]
        finite = values[np.isfinite(values)]
        if not finite.size:
            p_value = float("nan")
        else:
            lower_tail = (float(np.count_nonzero(finite <= 0.0)) + 1.0) / (len(finite) + 1.0)
            upper_tail = (float(np.count_nonzero(finite >= 0.0)) + 1.0) / (len(finite) + 1.0)
            p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
        lower, upper = _interval(values)
        result[name] = {
            "difference": float(full_candidate[name]) - float(full_reference[name]),
            "lower": lower,
            "upper": upper,
            "p_value": p_value,
            "iterations": iterations,
            "groups": len(keys),
        }
    return result


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
