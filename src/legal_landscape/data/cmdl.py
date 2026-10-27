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


def _flag(penalty: dict[str, Any], name: str) -> bool | None:
    if name not in penalty:
        return False
    value = penalty[name]
    if isinstance(value, bool):
        return value
    return None


class _UnusablePenalty(ValueError):
    """A placeholder or mistyped penalty field, not a negative month count."""


def _months(penalty: dict[str, Any], name: str) -> int | None:
    """Integer months, or ``None`` when the field is absent or a null placeholder.

    A non-integer value such as a blank string makes the whole object unusable.
    ``False`` is not a month count. Negative counts still raise.
    """
    if name not in penalty or penalty[name] is None or penalty[name] == "":
        return None
    value = penalty[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _UnusablePenalty(f"{name} must be an integer month count")
    if value < 0:
        raise ValueError(f"{name} months must be non-negative")
    return value


def _read_penalty(penalty: Any) -> tuple[str, int | None] | None:
    """Parse one penalty object. ``None`` means the object is missing or not usable.

    Exemption requires imprisonment, detention, and surveillance to all be the integer
    zero. ``{"imprisonment": None}``, ``{"imprisonment": ""}``, and
    ``{"death_penalty": False}`` are placeholders, not a real exemption.
    """
    if not isinstance(penalty, dict) or not (_PENALTY_FIELDS & set(penalty)):
        return None
    death = _flag(penalty, "death_penalty")
    life = _flag(penalty, "life_imprisonment")
    if death is None or life is None:
        return None
    try:
        imprisonment = _months(penalty, "imprisonment")
        detention = _months(penalty, "detention")
        surveillance = _months(penalty, "surveillance")
    except _UnusablePenalty:
        return None
    if death:
        return PenaltyType.DEATH.value, None
    if life:
        return PenaltyType.LIFE.value, None
    if imprisonment is not None and imprisonment > 0:
        return PenaltyType.FIXED_TERM.value, imprisonment
    if detention is not None and detention > 0:
        return PenaltyType.DETENTION.value, None
    if surveillance is not None and surveillance > 0:
        return PenaltyType.CONTROL.value, None
    if imprisonment == 0 and detention == 0 and surveillance == 0:
        return PenaltyType.EXEMPT.value, None
    return None


def _read_articles(judgments: Any) -> tuple[tuple[str, ...], bool]:
    """Article labels and whether every judgment actually supplied them.

    An empty list is a real annotation. A missing or null field is not.
    """
    if not isinstance(judgments, list) or not judgments:
        return (), False
    labels: list[Any] = []
    for judgment in judgments:
        if not isinstance(judgment, dict) or "article" not in judgment:
            return (), False
        if judgment.get("article") is None:
            return (), False
        labels.extend(judgment.get("article") or ())
    return normalize_criminal_articles(labels), True


def _normalize_outcome(
    outcome: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], str, int | None, bool, str | None, str | None, bool]:
    """Charges and articles come from every judgment. The executed penalty does not.

    A usable ``final_penalty`` is the executed sentence. One judgment without that
    field can supply its own penalty. Several judgments without it are not summed:
    the penalty label is unknown and must not be supervised. An unusable final
    penalty is not an exemption: a single judgment falls back, and several judgments
    stay unknown.
    """
    judgments = outcome.get("judgment") or []
    charges = tuple(
        str(item.get("standard_accusation") or item.get("accusation") or "").removesuffix("罪")
        for item in judgments
        if isinstance(item, dict)
    )
    charges = tuple(item for item in charges if item)
    articles, articles_available = _read_articles(judgments)
    final_raw = outcome.get("final_penalty")
    final = _read_penalty(final_raw)
    unusable_final = final_raw is not None and final is None
    if final is not None:
        penalty_type, months = final
        return charges, articles, penalty_type, months, False, None, None, articles_available
    if len(judgments) == 1 and isinstance(judgments[0], dict):
        single = _read_penalty(judgments[0].get("penalty"))
        if single is not None:
            penalty_type, months = single
            fallback = "unusable_final_penalty_fallback" if unusable_final else None
            return (
                charges, articles, penalty_type, months, False, None, fallback, articles_available
            )
        return (
            charges, articles, PenaltyType.UNKNOWN.value, None, True,
            "missing_penalty_fields", None, articles_available,
        )
    exclusion = "missing_final_penalty" if len(judgments) > 1 else "missing_penalty_fields"
    return (
        charges, articles, PenaltyType.UNKNOWN.value, None, True, exclusion, None,
        articles_available,
    )


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
                    (
                        charges,
                        articles,
                        penalty_type,
                        months,
                        mask_penalty,
                        exclusion,
                        fallback,
                        articles_available,
                    ) = _normalize_outcome(outcome)
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
                        articles_available=articles_available,
                        penalty_fallback=fallback,
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid CMDL record at {source}:{line_number}: {exc}") from exc
