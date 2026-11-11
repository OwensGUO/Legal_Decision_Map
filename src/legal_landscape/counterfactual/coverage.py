"""Coverage counts for parent cases and counterfactual generation.

Units are explicit. Parent length, charge, and defendant-count tables count defendant
records or cases, never a mixture of the two. Generation and view tables count
generation requests. A resumed run reports the full current view, not only the rows
written in that invocation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

LENGTH_BUCKETS = (
    ("lt_512", 0, 512),
    ("512_to_1023", 512, 1024),
    ("1024_to_2047", 1024, 2048),
    ("2048_to_4095", 2048, 4096),
    ("ge_4096", 4096, None),
)


def length_bucket(chars: int) -> str:
    for name, start, end in LENGTH_BUCKETS:
        if chars >= start and (end is None or chars < end):
            return name
    return "ge_4096"


def _counter(values: list[Any]) -> dict[str, int]:
    return dict(Counter(str(value) for value in values))


def _case_groups(parents: list[dict[str, Any]]) -> dict[str, int]:
    """One declared defendant-count per case group present in ``parents``.

    ``case_id`` is the defendant record. ``case_group_id`` is the case. A later record
    cannot shrink the declared size, so a co-defendant excluded from this table does not
    turn a two-defendant case into a one-defendant case.
    """
    groups: dict[str, int] = {}
    for parent in parents:
        group = parent.get("case_group_id")
        if not isinstance(group, str) or not group:
            raise ValueError("coverage parent is missing case_group_id")
        declared = int(parent["defendant_count"])
        groups[group] = max(groups.get(group, declared), declared)
    return groups


def _parent_table(parents: list[dict[str, Any]]) -> dict[str, Any]:
    charges: list[str] = []
    for parent in parents:
        charges.extend(str(charge) for charge in parent.get("charges") or ())
    groups = _case_groups(parents)
    return {
        "unit": "defendant_record",
        "count": len(parents),
        "case_count": len(groups),
        "length_buckets": _counter([length_bucket(int(parent["chars"])) for parent in parents]),
        "charges": _counter(charges),
        "defendants_per_case": {
            "unit": "case",
            "size": "declared_case_size",
            "counts": _counter(list(groups.values())),
            "rule": (
                "Each case_group_id in this table is one case. The bucket is the declared "
                "defendant count of that case, including co-defendants absent from this table."
            ),
        },
    }


def summarize_parents(
    parents: list[dict[str, Any]], rank_exclusions: dict[str, int]
) -> dict[str, Any]:
    """Distributions before and after the length filter, plus exclusion reasons."""
    kept = [parent for parent in parents if not parent.get("length_excluded")]
    length_excluded = [parent for parent in parents if parent.get("length_excluded")]
    kept_groups = set(_case_groups(kept))
    excluded_groups = set(_case_groups(length_excluded))
    return {
        "before_filters": _parent_table(parents),
        "after_filters": _parent_table(kept),
        "excluded": {
            "length": {
                "unit": "defendant_record",
                "count": len(length_excluded),
                "affected_cases": len(excluded_groups),
                "partial_cases": len(kept_groups & excluded_groups),
                "rule": (
                    "count is excluded defendant records. affected_cases is the number of "
                    "case groups that lost at least one defendant. partial_cases still have "
                    "another defendant record in after_filters."
                ),
            },
            "sentence_rank": {"unit": "defendant_record", "reasons": dict(rank_exclusions)},
        },
    }


def _validation_reasons(rows: list[dict[str, Any]]) -> dict[str, int]:
    reasons: Counter[str] = Counter()
    for row in rows:
        validation = row.get("validation") or {}
        if validation.get("valid") is True:
            continue
        errors = validation.get("errors") or ("invalid",)
        for error in errors:
            reasons[str(error)] += 1
    return dict(reasons)


def summarize_generation(
    records: list[dict[str, Any]],
    *,
    resumed: int,
    duplicate: int,
    transport_failures: int,
) -> dict[str, Any]:
    """Counts for this invocation. ``resumed`` rows are not part of ``records``."""
    passed = sum(bool((row.get("validation") or {}).get("valid")) for row in records)
    return {
        "unit": "generation_request",
        "new": len(records),
        "resumed": resumed,
        "duplicate": duplicate,
        "current_total": len(records) + resumed,
        "passed": passed,
        "failed_validation": len(records) - passed,
        "transport_failures": transport_failures,
        "excluded_reasons": _validation_reasons(records),
    }


def summarize_view(
    request_meta: dict[str, dict[str, Any]], view_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Every selected request in the published view, including rows reused from cache."""
    passed = sum(bool((row.get("validation") or {}).get("valid")) for row in view_rows)
    parents_by_defendant: dict[str, dict[str, Any]] = {}
    for row in view_rows:
        meta = request_meta.get(str(row.get("generation_id")))
        if meta is not None:
            parents_by_defendant[str(meta["case_id"])] = meta
    parents = list(parents_by_defendant.values())
    table = _parent_table(parents) if parents else {
        "unit": "defendant_record",
        "count": 0,
        "case_count": 0,
        "length_buckets": {},
        "charges": {},
        "defendants_per_case": {
            "unit": "case",
            "size": "declared_case_size",
            "counts": {},
        },
    }
    return {
        "unit": "generation_request",
        "records": len(view_rows),
        "passed": passed,
        "failed_validation": len(view_rows) - passed,
        "case_count": table["case_count"],
        "length_buckets": table["length_buckets"],
        "charges": table["charges"],
        "defendants_per_case": table["defendants_per_case"],
        "excluded_reasons": _validation_reasons(view_rows),
    }


def coverage_report(
    parents: list[dict[str, Any]],
    rank_exclusions: dict[str, int],
    records: list[dict[str, Any]],
    *,
    resumed: int,
    duplicate: int,
    transport_failures: int,
    request_meta: dict[str, dict[str, Any]] | None = None,
    view_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "units": {
            "parent_records": "defendant_record",
            "cases": "case",
            "generation_requests": "generation_request",
            "view_records": "generation_request",
        },
        "parents": summarize_parents(parents, rank_exclusions),
        "generation": summarize_generation(
            records,
            resumed=resumed,
            duplicate=duplicate,
            transport_failures=transport_failures,
        ),
    }
    if view_rows is not None:
        report["view"] = summarize_view(request_meta or {}, view_rows)
    return report
