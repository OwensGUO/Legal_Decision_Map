"""Streaming multi-defendant CMDL adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from legal_landscape.data.articles import normalize_criminal_articles
from legal_landscape.data.schema import CaseUnit, PenaltyType

_PENALTY_FIELDS = frozenset(
    {"death_penalty", "life_imprisonment", "imprisonment", "detention", "surveillance"}
)


def _read_penalty(penalty: Any) -> tuple[str, int | None] | None:
    """Parse one penalty object. ``None`` means the object is missing or not usable."""
    if not isinstance(penalty, dict) or not (_PENALTY_FIELDS & set(penalty)):
        return None
    if bool(penalty.get("death_penalty")):
        return PenaltyType.DEATH.value, None
    if bool(penalty.get("life_imprisonment")):
        return PenaltyType.LIFE.value, None
    months = int(penalty.get("imprisonment") or 0)
    if months < 0:
        raise ValueError("imprisonment months must be non-negative")
    if months > 0:
        return PenaltyType.FIXED_TERM.value, months
    detention = int(penalty.get("detention") or 0)
    if detention < 0:
        raise ValueError("detention months must be non-negative")
    if detention > 0:
        return PenaltyType.DETENTION.value, None
    surveillance = int(penalty.get("surveillance") or 0)
    if surveillance < 0:
        raise ValueError("surveillance months must be non-negative")
    if surveillance > 0:
        return PenaltyType.CONTROL.value, None
    return PenaltyType.EXEMPT.value, None


def _normalize_outcome(
    outcome: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], str, int | None, bool, str | None]:
    """Charges and articles come from every judgment. The executed penalty does not.

    A usable ``final_penalty`` is the executed sentence. One judgment without that
    field can supply its own penalty. Several judgments without it are not summed:
    the penalty label is unknown and must not be supervised.
    """
    judgments = outcome.get("judgment") or []
    charges = tuple(
        str(item.get("standard_accusation") or item.get("accusation") or "").removesuffix("罪")
        for item in judgments
        if isinstance(item, dict)
    )
    charges = tuple(item for item in charges if item)
    articles = normalize_criminal_articles(
        article
        for judgment in judgments
        if isinstance(judgment, dict)
        for article in (judgment.get("article") or ())
    )
    final = _read_penalty(outcome.get("final_penalty"))
    if final is not None:
        penalty_type, months = final
        return charges, articles, penalty_type, months, False, None
    if len(judgments) == 1 and isinstance(judgments[0], dict):
        single = _read_penalty(judgments[0].get("penalty"))
        if single is not None:
            penalty_type, months = single
            return charges, articles, penalty_type, months, False, None
        return charges, articles, PenaltyType.UNKNOWN.value, None, True, "missing_penalty_fields"
    exclusion = "missing_final_penalty" if len(judgments) > 1 else "missing_penalty_fields"
    return charges, articles, PenaltyType.UNKNOWN.value, None, True, exclusion


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
                names = [str(item) for item in row.get("defendants") or ()] or [
                    str(outcome["name"]) for outcome in outcomes
                ]
                for defendant_index, outcome in enumerate(outcomes):
                    charges, articles, penalty_type, months, mask_penalty, exclusion = (
                        _normalize_outcome(outcome)
                    )
                    name = str(outcome["name"])
                    yield CaseUnit(
                        dataset="cmdl",
                        split=split,
                        case_id=f"{group_id}-d{defendant_index:03d}",
                        group_id=group_id,
                        target_defendant=name,
                        fact_raw=str(row["fact"]),
                        charges=charges,
                        penalty_type=penalty_type,
                        imprisonment_months=months,
                        source_path=f"{source}:{line_number}",
                        conviction_articles=articles,
                        co_defendants=tuple(dict.fromkeys(item for item in names if item != name)),
                        mask_penalty=mask_penalty,
                        penalty_exclusion=exclusion,
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid CMDL record at {source}:{line_number}: {exc}") from exc
