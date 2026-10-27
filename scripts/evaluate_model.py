#!/usr/bin/env python3
"""Evaluate saved prediction JSONL without loading model weights."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

from legal_landscape.evaluation.bootstrap import (
    fast_bootstrap_metric_set,
    fast_paired_test,
    holm_adjust,
)
from legal_landscape.evaluation.prepared import (
    METRIC_VERSION,
    prepare_cmdl,
    prepare_counterfactual,
    prepare_static,
)
from legal_landscape.progress import create_progress_reporter

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
        "invariant_charge_drift",
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
    result.add_argument(
        "--parent-records",
        type=Path,
        help="Processed parent records (.jsonl or .jsonl.gz) to backfill parent charges in "
        "counterfactual rows written before parent_charge_indices existed",
    )
    result.add_argument("--metadata", type=Path, help="metadata.json with charge_vocabulary")
    result.add_argument("--dry-run", action="store_true")
    return result


def read_rows(
    path: Path, limit: int, *, complete_groups: bool = False
) -> list[dict[str, Any]]:
    """Read prediction rows.

    ``limit`` is a row cap. For CMDL, pass ``complete_groups=True`` so ``limit``
    counts cases and every defendant of a selected case is kept.
    """
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if not complete_groups:
        return rows[:limit]
    selected: list[str] = []
    for row in rows:
        group = str(row.get("group_id"))
        if group not in selected:
            if len(selected) >= limit:
                break
            selected.append(group)
    chosen = set(selected)
    return [row for row in rows if str(row.get("group_id")) in chosen]


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def backfill_parent_charges(
    rows: list[dict[str, Any]], parent_records: Path, metadata: Path
) -> None:
    """Add parent_charge_indices to flip rows from processed parent records."""
    needed = {
        str(row["case_id"])
        for row in rows
        if row.get("intervention_type") == "charge_flip" and not row.get("parent_charge_indices")
    }
    if not needed:
        return
    vocabulary = json.loads(metadata.read_text(encoding="utf-8"))["charge_vocabulary"]
    index = {label: position for position, label in enumerate(vocabulary)}
    charges: dict[str, list[int]] = {}
    with _open_text(parent_records) as handle:
        for line in handle:
            record = json.loads(line)
            if str(record.get("case_id")) in needed:
                charges[str(record["case_id"])] = [
                    index[label] for label in record.get("charges") or () if label in index
                ]
    for row in rows:
        if row.get("intervention_type") == "charge_flip" and not row.get("parent_charge_indices"):
            row["parent_charge_indices"] = charges.get(str(row["case_id"]), [])


def prepare(kind: str, rows: list[dict[str, Any]]) -> Any:
    if kind == "static":
        return prepare_static(rows)
    if kind == "counterfactual":
        return prepare_counterfactual(rows)
    if kind == "cmdl":
        return prepare_cmdl(rows)
    raise ValueError(f"unknown evaluation kind: {kind}")


def calculate_metrics(kind: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return prepare(kind, rows).metrics(None)


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


def _load(path: Path, args: argparse.Namespace) -> Any:
    rows = read_rows(path, args.limit, complete_groups=args.kind == "cmdl")
    missing_articles = [
        row for row in rows if row.get("articles_available") is False
    ]
    if missing_articles:
        raise SystemExit(
            f"refusing to score {len(missing_articles)} records with missing article labels; "
            "they are not all-negative article truth"
        )
    if args.kind == "counterfactual" and args.parent_records is not None:
        if args.metadata is None:
            raise SystemExit("--parent-records requires --metadata")
        backfill_parent_charges(rows, args.parent_records, args.metadata)
    if args.kind == "counterfactual":
        from legal_landscape.counterfactual.interventions import filter_sentence_rank_rows

        rows, exclusions = filter_sentence_rank_rows(rows)
    else:
        exclusions = {}
    try:
        prepared = prepare(args.kind, rows)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    prepared.sentence_rank_exclusions = exclusions
    prepared.source_rows = rows
    return prepared


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
    prepared = _load(args.input, args)
    metrics = prepared.metrics(None)
    available_bootstrap_names = tuple(
        name for name in bootstrap_names(args.kind) if name in metrics
    )
    with (
        create_progress_reporter() as progress,
        progress.task(f"Evaluating {args.kind} predictions", total=None),
    ):
        confidence_intervals = fast_bootstrap_metric_set(
            prepared,
            available_bootstrap_names,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
            progress=progress,
        )
        payload: dict[str, Any] = {
            **metrics,
            "metric_version": METRIC_VERSION,
            "sentence_rank_exclusions": getattr(prepared, "sentence_rank_exclusions", {}),
            "confidence_intervals": confidence_intervals,
            "bootstrap": {
                "iterations": args.bootstrap_iterations,
                "seed": args.bootstrap_seed,
                "cluster_key": "group_id",
            },
        }
        if args.reference_input is not None:
            reference = _load(args.reference_input, args)
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
            comparisons = fast_paired_test(
                prepared,
                reference,
                endpoint_names,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
                progress=progress,
                description="Paired bootstrap",
            )
            raw_p_values = {name: float(value["p_value"]) for name, value in comparisons.items()}
            finite_p_values = {
                name: value for name, value in raw_p_values.items() if math.isfinite(value)
            }
            adjusted = holm_adjust(finite_p_values)
            adjusted.update(
                {
                    name: float("nan")
                    for name, value in raw_p_values.items()
                    if not math.isfinite(value)
                }
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
