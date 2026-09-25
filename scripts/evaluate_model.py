#!/usr/bin/env python3
"""Evaluate saved prediction JSONL without loading model weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from legal_landscape.evaluation.bootstrap import (
    bootstrap_metric_set,
    holm_adjust,
    paired_cluster_test,
)
from legal_landscape.evaluation.cmdl_case_metrics import cmdl_metrics
from legal_landscape.evaluation.counterfactual_metrics import counterfactual_metrics
from legal_landscape.evaluation.static_metrics import (
    charge_metrics,
    multiclass_metrics,
    multilabel_metrics,
    sentence_class,
    sentence_metrics,
)

DEFAULT_PRIMARY_ENDPOINTS = {
    "static": (
        "charge_macro_f1",
        "article_macro_f1",
        "sentence_macro_f1",
        "sentence_mae",
        "sentence_tolerance_accuracy",
    ),
    "counterfactual": (
        "flip_accuracy",
        "invariant_violation_rate",
        "sentence_pairwise_accuracy",
        "monotonicity_violation_rate",
        "non_target_prediction_drift",
    ),
    "cmdl": ("defendant_accuracy", "case_exact_accuracy"),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", help="Reserved for experiment metadata")
    result.add_argument("--input", type=Path)
    result.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    result.add_argument("--kind", choices=("static", "counterfactual", "cmdl"), default="static")
    result.add_argument("--limit", type=int, default=1000)
    result.add_argument("--bootstrap-iterations", type=int, default=2000)
    result.add_argument("--bootstrap-seed", type=int, default=42)
    result.add_argument("--reference-input", type=Path)
    result.add_argument(
        "--primary-endpoints",
        help="Comma-separated scalar metric names; defaults to the declared endpoints",
    )
    result.add_argument("--dry-run", action="store_true")
    return result


def read_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for _, line in zip(range(limit), handle, strict=False)]


def calculate_metrics(kind: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("evaluation input contains no rows")
    if kind == "static":
        charge = charge_metrics(
            np.asarray([row["true_charges"] for row in rows]),
            np.asarray([row["predicted_charges"] for row in rows]),
        )
        sentence = sentence_metrics(
            np.asarray([row["predicted_months"] for row in rows]),
            np.asarray([row.get("true_months") for row in rows], dtype=float),
            [row["penalty_type"] for row in rows],
            charge_counts=np.asarray([sum(row["true_charges"]) for row in rows]),
        )
        true_sentence_classes = [
            sentence_class(row["penalty_type"], row.get("true_months")) for row in rows
        ]
        predicted_sentence_classes = [
            sentence_class(
                row.get("predicted_penalty_type", row["penalty_type"]),
                row.get("predicted_months"),
            )
            for row in rows
        ]
        sentence_classification = multiclass_metrics(
            true_sentence_classes,
            predicted_sentence_classes,
        )
        metrics: dict[str, Any] = {
            **{f"charge_{name}": value for name, value in charge.items()},
            **{
                f"sentence_{name}": value
                for name, value in sentence_classification.items()
            },
            **{f"sentence_{name}": value for name, value in sentence.items()},
            # Legacy aliases retained for existing result consumers.
            "macro_f1": charge["macro_f1"],
            "micro_f1": charge["micro_f1"],
            "exact_match": charge["exact_match"],
            **sentence,
        }
        if all("true_articles" in row and "predicted_articles" in row for row in rows):
            article = multilabel_metrics(
                np.asarray([row["true_articles"] for row in rows]),
                np.asarray([row["predicted_articles"] for row in rows]),
            )
            metrics.update({f"article_{name}": value for name, value in article.items()})
        return metrics
    if kind == "counterfactual":
        return counterfactual_metrics(rows)
    if kind == "cmdl":
        return cmdl_metrics(rows)
    raise ValueError(f"unknown evaluation kind: {kind}")


def bootstrap_names(kind: str) -> tuple[str, ...]:
    if kind == "static":
        return (
            "charge_accuracy",
            "charge_macro_precision",
            "charge_macro_recall",
            "charge_macro_f1",
            "article_accuracy",
            "article_macro_precision",
            "article_macro_recall",
            "article_macro_f1",
            "sentence_accuracy",
            "sentence_macro_precision",
            "sentence_macro_recall",
            "sentence_macro_f1",
            "sentence_mae",
            "sentence_log_mae",
            "sentence_tolerance_accuracy",
        )
    return DEFAULT_PRIMARY_ENDPOINTS[kind]


def main() -> int:
    args = parser().parse_args()
    if args.dry_run or args.input is None:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "kind": args.kind,
                    "limit": args.limit,
                    "bootstrap_iterations": args.bootstrap_iterations,
                    "bootstrap_seed": args.bootstrap_seed,
                    "reference_input": str(args.reference_input) if args.reference_input else None,
                }
            )
        )
        return 0
    rows = read_rows(args.input, args.limit)
    metrics = calculate_metrics(args.kind, rows)
    available_bootstrap_names = tuple(
        name for name in bootstrap_names(args.kind) if name in metrics
    )
    confidence_intervals = bootstrap_metric_set(
        rows,
        lambda sample: calculate_metrics(args.kind, sample),
        available_bootstrap_names,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    payload: dict[str, Any] = {
        **metrics,
        "confidence_intervals": confidence_intervals,
        "bootstrap": {
            "iterations": args.bootstrap_iterations,
            "seed": args.bootstrap_seed,
            "cluster_key": "group_id",
        },
    }
    if args.reference_input is not None:
        reference = read_rows(args.reference_input, args.limit)
        endpoint_names = (
            tuple(item.strip() for item in args.primary_endpoints.split(",") if item.strip())
            if args.primary_endpoints
            else tuple(
                name for name in DEFAULT_PRIMARY_ENDPOINTS[args.kind] if name in metrics
            )
        )
        if not endpoint_names or len(endpoint_names) > 5:
            raise ValueError("primary endpoints must contain between one and five metrics")
        unknown = [name for name in endpoint_names if name not in metrics]
        if unknown:
            raise ValueError(f"unknown primary endpoints: {', '.join(unknown)}")
        comparisons: dict[str, dict[str, float | int]] = {}
        raw_p_values: dict[str, float] = {}
        for name in endpoint_names:
            comparison = paired_cluster_test(
                rows,
                reference,
                lambda sample, metric_name=name: float(
                    calculate_metrics(args.kind, sample)[metric_name]
                ),
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            )
            comparisons[name] = comparison
            raw_p_values[name] = float(comparison["p_value"])
        finite_p_values = {
            name: value for name, value in raw_p_values.items() if math.isfinite(value)
        }
        adjusted = holm_adjust(finite_p_values)
        adjusted.update(
            {name: float("nan") for name, value in raw_p_values.items() if not math.isfinite(value)}
        )
        payload["comparison"] = {
            "reference_input": str(args.reference_input),
            "endpoints": comparisons,
            "raw_p_values": raw_p_values,
            "holm_adjusted_p_values": adjusted,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
