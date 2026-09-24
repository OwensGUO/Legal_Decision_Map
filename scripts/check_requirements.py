#!/usr/bin/env python3
"""Check imports, CUDA visibility, and optional local model configuration."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from legal_landscape.training.train import inspect_model_config

CORE = ("yaml", "numpy", "httpx")
TRAINING = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "safetensors",
    "sentencepiece",
)
INFERENCE = ("vllm",)
OPERATIONS = ("psutil", "pynvml", "tensorboard")
EXPECTED_GPU_VERSIONS = {
    "torch": "2.10.0",
    "transformers": "5.5.3",
    "accelerate": "1.13.0",
    "peft": "0.18.1",
    "bitsandbytes": "0.49.2",
    "vllm": "0.19.1",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", help="Optional YAML; retained for a uniform CLI")
    result.add_argument("--model-path", type=Path)
    result.add_argument("--cpu-only", action="store_true")
    result.add_argument("--limit", type=int, default=4)
    result.add_argument("--dry-run", action="store_true")
    return result


def _imports(names: tuple[str, ...]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            result[name] = str(getattr(module, "__version__", "imported"))
        except Exception as exc:  # Import-time binary errors are actionable here.
            result[name] = f"ERROR: {type(exc).__name__}: {exc}"
    return result


def has_requirement_failures(
    packages: dict[str, str | None], required: tuple[str, ...]
) -> bool:
    """Return whether any package requested for this invocation failed to import."""
    return any(
        str(packages.get(name, "ERROR: missing result")).startswith("ERROR")
        for name in required
    )


def version_mismatches(
    packages: dict[str, str | None], expected: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Return exact-version drift, allowing wheel-local suffixes such as +cu128."""
    mismatches: dict[str, dict[str, str]] = {}
    for name, expected_version in expected.items():
        actual = str(packages.get(name, "missing"))
        if actual.split("+", 1)[0] != expected_version:
            mismatches[name] = {"expected": expected_version, "actual": actual}
    return mismatches


def main() -> int:
    args = parser().parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "core": CORE,
                    "training": TRAINING,
                    "inference": INFERENCE,
                    "operations": OPERATIONS,
                }
            )
        )
        return 0
    required = CORE + (() if args.cpu_only else TRAINING + INFERENCE) + OPERATIONS
    packages = _imports(required)
    payload: dict[str, object] = {"packages": packages, "cpu_only": args.cpu_only}
    mismatches = {} if args.cpu_only else version_mismatches(packages, EXPECTED_GPU_VERSIONS)
    payload["version_mismatches"] = mismatches
    if args.model_path is not None:
        payload["model_config"] = inspect_model_config(args.model_path).__dict__
    if not args.cpu_only and not str(packages.get("torch", "")).startswith("ERROR"):
        import torch

        payload["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(i)
                for i in range(min(torch.cuda.device_count(), args.limit))
            ],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if has_requirement_failures(packages, required) or mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
