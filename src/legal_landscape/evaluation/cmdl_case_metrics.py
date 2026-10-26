"""CMDL defendant-, case-, and defendant-count-stratified metrics."""

from __future__ import annotations

from typing import Any

from legal_landscape.evaluation.prepared import prepare_cmdl


def cmdl_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("CMDL metrics require at least one row")
    return prepare_cmdl(rows).metrics(None)
