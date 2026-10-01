#!/usr/bin/env python3
"""Validate or initialize generator provenance before generation and downstream resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_landscape.config import load_config, parse_overrides
from legal_landscape.counterfactual.provenance import ensure_manifest, generator_identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, action="append", default=[])
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = load_config(args.config, overrides=parse_overrides(args.set))["generator"]
    try:
        identity = generator_identity(config, mock=args.mock)
        ensure_manifest(
            args.manifest,
            identity,
            artifact_roots=[
                args.manifest.parent,
                *args.artifact_root,
            ],
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(identity, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
