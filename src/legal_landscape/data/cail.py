"""Streaming CAIL2018 adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from legal_landscape.data.schema import CaseUnit, PenaltyType


def _penalty(term: dict[str, Any]) -> tuple[str, int | None]:
    if bool(term.get("death_penalty")):
        return PenaltyType.DEATH.value, None
    if bool(term.get("life_imprisonment")):
        return PenaltyType.LIFE.value, None
    months = int(term.get("imprisonment") or 0)
    if months <= 0:
        return PenaltyType.EXEMPT.value, None
    return PenaltyType.FIXED_TERM.value, months


def iter_cail(path: str | Path, *, split: str, limit: int | None = None) -> Iterator[CaseUnit]:
    """Yield one normalized case per CAIL JSONL record."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and line_number > limit:
                break
            try:
                row = json.loads(line)
                meta = row["meta"]
                defendants = tuple(meta.get("criminals") or ())
                if len(defendants) != 1:
                    raise ValueError("CAIL-small requires exactly one defendant")
                penalty_type, months = _penalty(meta.get("term_of_imprisonment") or {})
                canonical = json.dumps(
                    {"fact": row["fact"], "criminals": defendants},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
                identity = f"cail-{digest}"
                yield CaseUnit(
                    dataset="cail",
                    split=split,
                    case_id=identity,
                    group_id=identity,
                    target_defendant=str(defendants[0]),
                    fact_raw=str(row["fact"]),
                    charges=tuple(str(item).removesuffix("罪") for item in meta["accusation"]),
                    penalty_type=penalty_type,
                    imprisonment_months=months,
                    source_path=f"{source}:{line_number}",
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid CAIL record at {source}:{line_number}: {exc}") from exc
