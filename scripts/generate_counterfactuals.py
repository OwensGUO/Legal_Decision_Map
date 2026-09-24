#!/usr/bin/env python3
"""Generate typed counterfactuals with mock or local vLLM backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from legal_landscape.config import load_config, parse_overrides
from legal_landscape.counterfactual.generate import (
    GenerationRequest,
    MockGenerator,
    VLLMHTTPGenerator,
    generate_records,
)
from legal_landscape.counterfactual.interventions import propose_interventions
from legal_landscape.counterfactual.prompts import PROMPT_VERSION
from legal_landscape.factors.schema import InterventionSpec, LegalFactors


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/cf/qwen36_27b.yaml")
    result.add_argument("--input", type=Path)
    result.add_argument("--output", type=Path, default=Path("outputs/counterfactuals.jsonl"))
    result.add_argument("--limit", type=int, default=10)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "--execute", action="store_true", help="Allow calls to the configured vLLM service"
    )
    result.add_argument("--mock", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return result


def _requests(
    path: Path,
    limit: int,
    boundaries: tuple[tuple[str, ...], ...],
    *,
    max_source_chars: int | None = None,
) -> tuple[list[GenerationRequest], int]:
    """Build at most ``limit`` requests; cases longer than ``max_source_chars`` are skipped."""
    requests: list[GenerationRequest] = []
    skipped_long = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "spec" in row:
                requests.append(
                    GenerationRequest(row["parent_text"], InterventionSpec.from_dict(row["spec"]))
                )
            else:
                parent_text = row.get("fact_conservative") or row["fact_raw"]
                if max_source_chars is not None and len(parent_text) > max_source_chars:
                    skipped_long += 1
                    continue
                factors = LegalFactors.from_dict(row["factors"])
                for spec in propose_interventions(
                    parent_case_id=row["case_id"],
                    defendant=row["target_defendant"],
                    charge=row["charges"][0],
                    factors=factors,
                    charge_boundaries=boundaries,
                ):
                    requests.append(GenerationRequest(parent_text, spec))
                    if len(requests) >= limit:
                        break
            if len(requests) >= limit:
                break
    return requests, skipped_long


def main() -> int:
    args = parser().parse_args()
    loaded = load_config(args.config, overrides=parse_overrides(args.set))
    config = loaded["generator"]
    boundaries = tuple(tuple(group) for group in loaded.get("charge_boundaries", ()))
    summary = {
        "backend": "mock" if args.mock else config["backend"],
        "limit": args.limit,
        "output": str(args.output),
    }
    if args.dry_run or not args.execute:
        print(json.dumps({**summary, "dry_run": True}, ensure_ascii=False, indent=2))
        return 0
    if args.input is None:
        raise SystemExit("--input is required with --execute")
    sampling = {key: config[key] for key in ("temperature", "top_p", "max_tokens") if key in config}
    concurrency = int(config.get("concurrency", 1))
    generator = (
        MockGenerator()
        if args.mock
        else VLLMHTTPGenerator(
            config["endpoint"],
            model=config["model_path"],
            timeout=float(config.get("request_timeout", 600)),
            sampling=sampling,
            allow_remote=bool(config.get("allow_remote", False)),
            max_connections=concurrency,
            request_retries=int(config.get("request_retries", 3)),
        )
    )
    max_source_chars = config.get("max_source_chars")
    requests, skipped_long = _requests(
        args.input,
        args.limit,
        boundaries,
        max_source_chars=int(max_source_chars) if max_source_chars is not None else None,
    )
    try:
        result = generate_records(
            requests,
            generator,
            args.output,
            model_revision=config.get("model_revision", "local"),
            prompt_version=PROMPT_VERSION,
            sampling=sampling,
            seed=int(config.get("seed", 42)),
            retries=int(config.get("retries", 2)),
            resume=args.resume,
            min_similarity=float(config.get("min_similarity", 0.5)),
            concurrency=concurrency,
        )
    finally:
        if isinstance(generator, VLLMHTTPGenerator):
            generator.close()
    valid = sum(bool(record["validation"]["valid"]) for record in result.records)
    report = {
        **summary,
        "requests": len(requests),
        "skipped_long_cases": skipped_long,
        "skipped_completed": result.skipped_completed,
        "skipped_duplicate": result.skipped_duplicate,
        "generated": len(result.records),
        "valid": valid,
        "transport_failures": len(result.failed),
    }
    print(json.dumps(report, ensure_ascii=False))
    if result.failed:
        print(
            json.dumps({"failed_examples": result.failed[:5]}, ensure_ascii=False),
            file=sys.stderr,
        )
        print("Some requests failed; rerun with --resume to retry them.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
