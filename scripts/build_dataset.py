#!/usr/bin/env python3
"""Build bounded, traceable processed data from read-only source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_landscape.config import parse_overrides
from legal_landscape.data.build import build_dataset


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", required=True)
    result.add_argument("--output-dir", type=Path, default=Path("outputs/processed"))
    result.add_argument("--limit", type=int, default=100)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.dry_run or not args.execute:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "config": args.config,
                    "output_dir": str(args.output_dir),
                    "limit": args.limit,
                },
                ensure_ascii=False,
            )
        )
        return 0
    summary = build_dataset(
        args.config,
        args.output_dir,
        limit=args.limit,
        overrides=parse_overrides(args.set),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
