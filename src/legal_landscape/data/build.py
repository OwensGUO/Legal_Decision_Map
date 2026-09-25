"""Build traceable processed JSONL without modifying source datasets."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from legal_landscape.config import load_config
from legal_landscape.data.cail import iter_cail
from legal_landscape.data.cmdl import iter_cmdl
from legal_landscape.data.manifest import build_manifest
from legal_landscape.data.sanitize import sanitize_text
from legal_landscape.factors.extract import extract_factors

_SPLIT_PRIORITY = {"train": 0, "valid": 1, "test": 2}


def _source_paths(data: dict[str, Any]) -> dict[str, Path]:
    root = Path(data["root"])
    if data["dataset"] == "cail":
        directory = root / data.get("variant", "exercise_contest")
        return {
            "train": directory / "data_train.json",
            "valid": directory / "data_valid.json",
            "test": directory / "data_test.json",
        }
    if data["dataset"] == "cmdl":
        directory = root / data.get("variant", "small")
        return {
            "train": directory / "train_small.jsonl",
            "valid": directory / "valid_small.jsonl",
            "test": directory / "test_small.jsonl",
        }
    raise ValueError(f"unsupported dataset: {data['dataset']}")


def _iter_cases(
    data: dict[str, Any], source: Path, *, split: str, limit: int | None
):
    if data["dataset"] == "cail":
        return iter_cail(source, split=split, limit=limit)
    return iter_cmdl(source, split=split, limit=limit)


def _assign_groups_to_splits(
    data: dict[str, Any], paths: dict[str, Path], *, limit: int | None
) -> tuple[dict[str, str], set[str]]:
    """Assign duplicate source groups to the most held-out official split."""
    assignments: dict[str, str] = {}
    cross_split_groups: set[str] = set()
    for split, source in paths.items():
        for case in _iter_cases(data, source, split=split, limit=limit):
            previous = assignments.setdefault(case.group_id, split)
            if previous == split:
                continue
            cross_split_groups.add(case.group_id)
            if _SPLIT_PRIORITY[split] > _SPLIT_PRIORITY[previous]:
                assignments[case.group_id] = split
    return assignments, cross_split_groups


def build_dataset(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stream all configured splits into auditable processed records."""
    config = load_config(config_path, overrides=overrides)
    data = config["data"]
    paths = _source_paths(data)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    split_units: dict[str, int] = {}
    charges: set[str] = set()
    articles: set[str] = set()
    group_splits, cross_split_groups = _assign_groups_to_splits(
        data, paths, limit=limit
    )
    dropped_units = dict.fromkeys(paths, 0)
    for split, source in paths.items():
        iterator = _iter_cases(data, source, split=split, limit=limit)
        count = 0
        output_path = destination / f"{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for case in iterator:
                if group_splits[case.group_id] != split:
                    dropped_units[split] += 1
                    continue
                sanitized = sanitize_text(case.fact_raw, charges=case.charges)
                factors = extract_factors(case.fact_raw, target_defendant=case.target_defendant)
                record = {
                    **case.to_dict(),
                    "fact_conservative": sanitized.conservative,
                    "fact_strict": sanitized.strict,
                    "redactions": [asdict(item) for item in sanitized.redactions],
                    "factors": factors.to_dict(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                charges.update(case.charges)
                articles.update(case.conviction_articles)
                count += 1
        split_units[split] = count
    manifest = [entry.to_dict() for entry in build_manifest(list(paths.values()))]
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "dataset": data["dataset"],
        "split_units": split_units,
        "split_integrity": {
            "assignment_policy": "prefer_test_then_valid_then_train",
            "cross_split_groups": len(cross_split_groups),
            "dropped_units": dropped_units,
        },
        "charge_vocabulary": sorted(charges),
        "article_vocabulary": sorted(articles),
        "penalty_vocabulary": [
            "fixed_term",
            "life",
            "death",
            "detention",
            "control",
            "exempt",
            "unknown",
        ],
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
