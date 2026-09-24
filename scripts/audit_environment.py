#!/usr/bin/env python3
"""Report Python, operating-system, CUDA, GPU, and optional runtime details."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", help="Optional model YAML to report")
    result.add_argument("--limit", type=int, default=4, help="Maximum GPUs to report")
    result.add_argument("--dry-run", action="store_true")
    return result


def _os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {"system": platform.system(), "release": platform.release()}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.lower()] = value.strip().strip('"')
    return result


def audit(limit: int) -> dict[str, Any]:
    visible_devices = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    payload: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "os_release": _os_release(),
        "cuda_visible_devices": visible_devices,
        "cuda": {"available": False, "device_count": 0, "devices": []},
    }
    try:
        import torch

        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
        payload["torch"] = torch.__version__
        payload["cuda"] = {
            "available": available,
            "compiled_version": torch.version.cuda,
            "device_count": count,
            "devices": [torch.cuda.get_device_name(index) for index in range(min(count, limit))],
        }
    except ImportError:
        payload["torch"] = None
    try:
        import psutil

        payload["memory_bytes"] = psutil.virtual_memory().total
        payload["cpu_count"] = psutil.cpu_count(logical=True)
    except ImportError:
        payload["psutil"] = None
    try:
        import pynvml

        pynvml.nvmlInit()
        payload["nvidia_driver"] = pynvml.nvmlSystemGetDriverVersion()
        payload["nvml_device_count"] = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
    except Exception as exc:
        payload["nvml"] = f"unavailable: {type(exc).__name__}"
    return payload


def main() -> int:
    args = parser().parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "checks": ["python", "os", "torch", "cuda", "memory", "nvml"],
                }
            )
        )
        return 0
    print(json.dumps(audit(args.limit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
