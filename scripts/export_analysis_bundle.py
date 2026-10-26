#!/usr/bin/env python3
"""Package evaluation, training, and runtime evidence from OUTPUT_ROOT for offline analysis.

Read-only for OUTPUT_ROOT. Uses only the standard library so it runs with any project
version. Checkpoint weights, optimizer states, TensorBoard events, and the full
counterfactual caches are skipped; processed data and counterfactuals are slimmed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tarfile
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_KEYS = (
    "case_id", "group_id", "source_path", "target_defendant", "charges",
    "conviction_articles", "penalty_type", "imprisonment_months", "factors",
)
CF_KEYS = (
    "generation_id", "parent_case_id", "intervention_type", "target_charge",
    "rank_direction", "rule_id", "retry_count", "changed_fields",
)
LOG_TAIL_LINES = 3000
LARGE_LOG_BYTES = 200 * 1024 * 1024


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "outputs",
        help="The OUTPUT_ROOT used by run.sh (default: <project>/outputs)",
    )
    result.add_argument("--dest", type=Path, default=Path.cwd(), help="Directory for the archive")
    result.add_argument("--generator", default="qwen38")
    result.add_argument("--mode", default="main", help="Counterfactual view mode to slim")
    result.add_argument("--sample", type=int, default=3000, help="Full counterfactual samples")
    result.add_argument(
        "--console-log", type=Path, action="append", default=[],
        help="Saved run.sh terminal log; *.log files in the project root are added too",
    )
    result.add_argument("--include-smoke", action="store_true")
    return result


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl_gz(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=lines))


def _copy(source: Path, target: Path, missing: list[str]) -> None:
    if not source.is_file():
        missing.append(str(source))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _export_processed(out: Path, stage: Path, missing: list[str]) -> None:
    for dataset in ("cail_small", "cmdl_small"):
        directory = out / "processed" / dataset
        for split in ("train", "valid", "test"):
            source = directory / f"{split}.jsonl"
            if not source.is_file():
                missing.append(str(source))
                continue
            keys = LABEL_KEYS + (("fact_strict",) if split == "test" else ())
            _write_jsonl_gz(
                stage / "processed" / f"{dataset}-{split}-labels.jsonl.gz",
                ({key: row.get(key) for key in keys} for row in _read_jsonl(source)),
            )
        for name in ("metadata.json", "manifest.json"):
            _copy(directory / name, stage / "processed" / f"{dataset}-{name}", missing)


def _slim_and_sample(
    source: Path, sample: list[dict[str, Any]], sample_size: int, rng: random.Random
):
    """Yield slim rows for every record while reservoir-sampling full records into ``sample``."""
    for index, row in enumerate(_read_jsonl(source)):
        if len(sample) < sample_size:
            sample.append(row)
        else:
            slot = rng.randrange(index + 1)
            if slot < sample_size:
                sample[slot] = row
        validation = row.get("validation") or {}
        slim = {key: row.get(key) for key in CF_KEYS}
        slim.update(
            valid=validation.get("valid"),
            errors=validation.get("errors"),
            leakage=row.get("leakage"),
        )
        yield slim


def _export_counterfactuals(
    out: Path, stage: Path, generator: str, mode: str, sample_size: int, missing: list[str]
) -> None:
    namespace = out / "counterfactuals" / generator
    _copy(namespace / "generator-provenance.json",
          stage / "counterfactuals" / "generator-provenance.json", missing)
    rng = random.Random(42)
    views = (("", namespace / "views" / mode), ("heldout-", namespace / "heldout/views" / mode))
    for prefix, view_root in views:
        for dataset in ("cail", "cmdl"):
            source = view_root / f"{dataset}.jsonl"
            if not source.is_file():
                missing.append(str(source))
                continue
            sample: list[dict[str, Any]] = []
            _write_jsonl_gz(
                stage / "counterfactuals" / f"cf-{prefix}{dataset}-all-slim.jsonl.gz",
                _slim_and_sample(source, sample, sample_size, rng),
            )
            split = "test" if prefix else "train"
            parents_path = out / "processed" / f"{dataset}_small" / f"{split}.jsonl"
            wanted = {row.get("parent_case_id") for row in sample}
            parents: dict[str, Any] = {}
            if parents_path.is_file():
                for row in _read_jsonl(parents_path):
                    if row.get("case_id") in wanted:
                        parents[row["case_id"]] = {
                            key: row.get(key)
                            for key in ("fact_conservative", "fact_strict", "charges")
                        }
            _write_jsonl_gz(
                stage / "counterfactuals" / f"cf-{prefix}{dataset}-sample.jsonl.gz",
                ({**row, "parent": parents.get(row.get("parent_case_id"))} for row in sample),
            )


def _run_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if "tensorboard" in parts:
        return None
    if any(part.startswith("checkpoint-") for part in parts):
        if info.isdir() or parts[-1] == "training_progress.json":
            return info
        return None
    return info


def _run_directories(out: Path, generator: str, include_smoke: bool) -> list[Path]:
    root = out / "runs" / generator
    if not root.is_dir():
        return []
    return sorted(
        child for child in root.iterdir()
        if child.is_dir() and (include_smoke or child.name != "smoke")
    )


def _checkpoints(directories: list[Path], out: Path) -> list[dict[str, Any]]:
    found = []
    for directory in directories:
        for checkpoint in sorted(directory.rglob("checkpoint-*")):
            if not checkpoint.is_dir():
                continue
            progress = checkpoint / "training_progress.json"
            steps = None
            if progress.is_file():
                try:
                    steps = json.loads(progress.read_text(encoding="utf-8")).get("steps")
                except (OSError, ValueError):
                    steps = None
            found.append({
                "path": str(checkpoint.relative_to(out)),
                "modified": datetime.fromtimestamp(
                    checkpoint.stat().st_mtime, UTC
                ).isoformat(),
                "steps": steps,
            })
    return found


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _disk_usage(out: Path, generator: str) -> dict[str, int]:
    usage: dict[str, int] = {}
    for pattern in (f"runs/{generator}/*/*/*", f"counterfactuals/{generator}/*", "processed/*"):
        for path in sorted(out.glob(pattern)):
            usage[str(path.relative_to(out))] = (
                _directory_bytes(path) if path.is_dir() else path.stat().st_size
            )
    return usage


def _environment(stage: Path) -> None:
    target = stage / "environment"
    target.mkdir(parents=True, exist_ok=True)
    for name, command in (
        ("pip-freeze.txt", [sys.executable, "-m", "pip", "freeze"]),
        ("nvidia-smi.txt", ["nvidia-smi"]),
    ):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            (target / name).write_text(result.stdout + result.stderr, encoding="utf-8")
        except (OSError, subprocess.SubprocessError) as exc:
            (target / name).write_text(f"unavailable: {exc}\n", encoding="utf-8")
    code = stage / "code"
    for relative in ("run.sh", "requirements.txt"):
        if (PROJECT_ROOT / relative).is_file():
            code.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT_ROOT / relative, code / relative)
    if (PROJECT_ROOT / "configs").is_dir():
        shutil.copytree(PROJECT_ROOT / "configs", code / "configs", dirs_exist_ok=True)
    hashes = {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("src", "scripts")
        if (PROJECT_ROOT / folder).is_dir()
        for path in sorted((PROJECT_ROOT / folder).rglob("*.py"))
    }
    if hashes:
        code.mkdir(parents=True, exist_ok=True)
        (code / "source-sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def _logs(out: Path, stage: Path, console_logs: list[Path], missing: list[str]) -> None:
    for name in ("run-summary.json", "gpu-probe.json"):
        _copy(out / name, stage / "meta" / name, missing)
    _copy(out / "logs" / "vllm-probe.json", stage / "meta" / "vllm-probe.json", missing)
    vllm_log = out / "logs" / "vllm.log"
    if vllm_log.is_file():
        (stage / "logs").mkdir(parents=True, exist_ok=True)
        (stage / "logs" / "vllm-tail.log").write_text(_tail(vllm_log), encoding="utf-8")
    else:
        missing.append(str(vllm_log))
    candidates = list(console_logs) + sorted(PROJECT_ROOT.glob("*.log"))
    for path in dict.fromkeys(candidate.resolve() for candidate in candidates):
        if not path.is_file():
            missing.append(str(path))
            continue
        target = stage / "console" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.stat().st_size > LARGE_LOG_BYTES:
            target.write_text(_tail(path, 200000), encoding="utf-8")
        else:
            shutil.copy2(path, target)


def main() -> int:
    args = parser().parse_args()
    out = args.output_root.resolve()
    if not out.is_dir():
        print(f"OUTPUT_ROOT not found: {out}; pass --output-root", file=sys.stderr)
        return 2
    started = time.time()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"analysis-bundle-{stamp}"
    args.dest.mkdir(parents=True, exist_ok=True)
    stage = args.dest / f".{name}-staging"
    archive = args.dest / f"{name}.tar.gz"
    missing: list[str] = []
    try:
        stage.mkdir(parents=True)
        print("[1/5] processed labels", file=sys.stderr)
        _export_processed(out, stage, missing)
        print("[2/5] counterfactual summaries and samples", file=sys.stderr)
        _export_counterfactuals(out, stage, args.generator, args.mode, args.sample, missing)
        print("[3/5] logs, environment, and code fingerprints", file=sys.stderr)
        _logs(out, stage, args.console_log, missing)
        _environment(stage)
        print("[4/5] checkpoint timeline and disk usage", file=sys.stderr)
        run_directories = _run_directories(out, args.generator, args.include_smoke)
        if not run_directories:
            missing.append(str(out / "runs" / args.generator))
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "output_root": str(out),
            "generator": args.generator,
            "mode": args.mode,
            "sample": args.sample,
            "checkpoints": _checkpoints(run_directories, out),
            "disk_usage_bytes": _disk_usage(out, args.generator),
            "missing": missing,
        }
        (stage / "bundle-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("[5/5] archive", file=sys.stderr)
        with tarfile.open(archive, "w:gz") as bundle:
            for item in sorted(stage.iterdir()):
                bundle.add(item, arcname=f"{name}/{item.name}")
            for directory in run_directories:
                bundle.add(
                    directory,
                    arcname=f"{name}/{directory.relative_to(out)}",
                    filter=_run_filter,
                )
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    size = archive.stat().st_size / 1024 / 1024
    print(json.dumps({
        "archive": str(archive.resolve()),
        "size_mb": round(size, 1),
        "missing": len(missing),
        "seconds": round(time.time() - started, 1),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
