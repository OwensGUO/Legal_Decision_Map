#!/usr/bin/env python3
"""Execute bounded CUDA, BF16, NF4, and optional Triton compatibility probes."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import Any


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--limit", type=int, default=4, help="Required visible CUDA devices")
    result.add_argument(
        "--probe-fla", action="store_true", help="Also import FLA and launch Triton"
    )
    result.add_argument("--dry-run", action="store_true")
    return result


def _probe_triton(torch: Any) -> str:
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(first, second, output, size: tl.constexpr):
        offsets = tl.arange(0, size)
        tl.store(output + offsets, tl.load(first + offsets) + tl.load(second + offsets))

    first = torch.arange(64, device="cuda", dtype=torch.float32)
    second = torch.ones_like(first)
    output = torch.empty_like(first)
    add_kernel[(1,)](first, second, output, size=64)
    torch.cuda.synchronize()
    if not torch.equal(output, first + second):
        raise RuntimeError("Triton probe returned incorrect values")
    return str(getattr(triton, "__version__", "imported"))


def probe(required_devices: int, *, probe_fla: bool) -> tuple[dict[str, Any], bool]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": [
            item.strip()
            for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if item.strip()
        ],
        "required_devices": required_devices,
        "errors": [],
    }
    errors: list[str] = payload["errors"]
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["torch_cuda_runtime"] = torch.version.cuda
        payload["cuda_available"] = torch.cuda.is_available()
        payload["visible_device_count"] = torch.cuda.device_count()
        payload["devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if not torch.cuda.is_available():
            errors.append("CUDA is unavailable")
        elif torch.cuda.device_count() != required_devices:
            errors.append(
                f"expected {required_devices} visible CUDA devices, got {torch.cuda.device_count()}"
            )
        else:
            matrix = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
            result = matrix @ matrix
            torch.cuda.synchronize()
            if float(result.float().sum()) != 64.0**3:
                errors.append("BF16 CUDA matrix probe returned an incorrect result")
            try:
                import bitsandbytes as bnb

                layer = bnb.nn.Linear4bit(
                    64,
                    64,
                    bias=False,
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
                ).to("cuda")
                nf4_result = layer(matrix)
                torch.cuda.synchronize()
                payload["bitsandbytes"] = str(getattr(bnb, "__version__", "imported"))
                payload["nf4_shape"] = list(nf4_result.shape)
            except Exception as exc:
                errors.append(f"NF4 probe failed: {type(exc).__name__}: {exc}")
            if probe_fla:
                try:
                    import fla

                    payload["flash_linear_attention"] = str(
                        getattr(fla, "__version__", "imported")
                    )
                    payload["triton"] = _probe_triton(torch)
                except Exception as exc:
                    errors.append(f"FLA/Triton probe failed: {type(exc).__name__}: {exc}")
    except Exception as exc:
        errors.append(f"PyTorch CUDA probe failed: {type(exc).__name__}: {exc}")
    return payload, not errors


def main() -> int:
    args = parser().parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "required_devices": args.limit,
                    "probe_fla": args.probe_fla,
                    "checks": ["CUDA", "BF16", "bitsandbytes NF4", "optional FLA/Triton"],
                }
            )
        )
        return 0
    payload, passed = probe(args.limit, probe_fla=args.probe_fla)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
