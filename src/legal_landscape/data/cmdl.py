"""Streaming multi-defendant CMDL adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from legal_landscape.data.articles import normalize_criminal_articles
from legal_landscape.data.schema import CaseUnit, PenaltyType


def _normalize_outcome(
    outcome: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], str, int | None]:
    judgments = outcome.get("judgment") or []
    charges = tuple(
        str(item.get("standard_accusation") or item.get("accusation") or "").removesuffix("罪")
        for item in judgments
    )
    charges = tuple(item for item in charges if item)
    articles = normalize_criminal_articles(
        article
        for judgment in judgments
        for article in (judgment.get("article") or ())
    )
    penalties = [item.get("penalty") or {} for item in judgments]
    if any(bool(item.get("death_penalty")) for item in penalties):
        return charges, articles, PenaltyType.DEATH.value, None
    if any(bool(item.get("life_imprisonment")) for item in penalties):
        return charges, articles, PenaltyType.LIFE.value, None
    # Criminal Law art. 69: fixed-term imprisonment absorbs concurrent criminal detention.
    months = sum(int(item.get("imprisonment") or 0) for item in penalties)
    if months > 0:
        return charges, articles, PenaltyType.FIXED_TERM.value, months
    if any(int(item.get("detention") or 0) > 0 for item in penalties):
        return charges, articles, PenaltyType.DETENTION.value, None
    if any(int(item.get("surveillance") or 0) > 0 for item in penalties):
        return charges, articles, PenaltyType.CONTROL.value, None
    return charges, articles, PenaltyType.EXEMPT.value, None


def iter_cmdl(path: str | Path, *, split: str, limit: int | None = None) -> Iterator[CaseUnit]:
    """Yield a target-defendant unit for every labelled CMDL outcome."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and line_number > limit:
                break
            try:
                row = json.loads(line)
                outcomes = row.get("outcomes") or []
                if not outcomes:
                    raise ValueError("CMDL record has no defendant outcomes")
                canonical = json.dumps(
                    {
                        "fact": row["fact"],
                        "defendants": row.get("defendants")
                        or [outcome.get("name") for outcome in outcomes],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
                group_id = f"cmdl-{digest}"
                for defendant_index, outcome in enumerate(outcomes):
                    charges, articles, penalty_type, months = _normalize_outcome(outcome)
                    yield CaseUnit(
                        dataset="cmdl",
                        split=split,
                        case_id=f"{group_id}-d{defendant_index:03d}",
                        group_id=group_id,
                        target_defendant=str(outcome["name"]),
                        fact_raw=str(row["fact"]),
                        charges=charges,
                        penalty_type=penalty_type,
                        imprisonment_months=months,
                        source_path=f"{source}:{line_number}",
                        conviction_articles=articles,
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid CMDL record at {source}:{line_number}: {exc}") from exc
