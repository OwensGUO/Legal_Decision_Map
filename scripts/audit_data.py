#!/usr/bin/env python3
"""Audit source data paths, schemas, and manifests without modifying them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_landscape.config import load_config, parse_overrides
from legal_landscape.data.cail import iter_cail
from legal_landscape.data.cmdl import iter_cmdl
from legal_landscape.data.manifest import build_manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--limit", type=int, default=10)
    result.add_argument(
        "--dry-run", action="store_true", help="Report only; audit is always read-only"
    )
    result.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return result


def main() -> int:
    args = parser().parse_args()
    config = load_config(args.config, overrides=parse_overrides(args.set))
    data = config["data"]
    root = Path(data["root"])
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "dataset": data["dataset"],
                    "root": str(root),
                    "limit": args.limit,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if data["dataset"] == "cail":
        names = {"train": "data_train.json", "valid": "data_valid.json", "test": "data_test.json"}
        paths = {
            key: root / data.get("variant", "exercise_contest") / name
            for key, name in names.items()
        }
        counts = {
            key: sum(1 for _ in iter_cail(path, split=key, limit=args.limit))
            for key, path in paths.items()
        }
    elif data["dataset"] == "cmdl":
        names = {
            "train": "train_small.jsonl",
            "valid": "valid_small.jsonl",
            "test": "test_small.jsonl",
        }
        paths = {key: root / data.get("variant", "small") / name for key, name in names.items()}
        counts = {
            key: sum(1 for _ in iter_cmdl(path, split=key, limit=args.limit))
            for key, path in paths.items()
        }
    else:
        raise ValueError(f"unsupported dataset: {data['dataset']}")
    payload = {
        "dataset": data["dataset"],
        "sample_units": counts,
        "manifest": [entry.to_dict() for entry in build_manifest(list(paths.values()))],
        "cmdl_big_eligible": (
            bool(list(root.glob("train_big*.jsonl")))
            and bool(list(root.glob("valid_big*.jsonl")))
            and bool(list(root.glob("test_big*.jsonl")))
            if data["dataset"] == "cmdl"
            else None
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
