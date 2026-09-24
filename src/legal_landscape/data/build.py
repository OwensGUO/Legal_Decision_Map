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
    group_splits: dict[str, str] = {}
    for split, source in paths.items():
        iterator = (
            iter_cail(source, split=split, limit=limit)
            if data["dataset"] == "cail"
            else iter_cmdl(source, split=split, limit=limit)
        )
        count = 0
        output_path = destination / f"{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for case in iterator:
                previous = group_splits.setdefault(case.group_id, split)
                if previous != split:
                    raise ValueError(f"group {case.group_id} crosses splits {previous}/{split}")
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
                count += 1
        split_units[split] = count
    manifest = [entry.to_dict() for entry in build_manifest(list(paths.values()))]
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "dataset": data["dataset"],
        "split_units": split_units,
        "charge_vocabulary": sorted(charges),
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
