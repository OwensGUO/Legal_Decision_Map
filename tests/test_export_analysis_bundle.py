from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/export_analysis_bundle.py"


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _output_root(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    for dataset in ("cail_small", "cmdl_small"):
        for split in ("train", "valid", "test"):
            _jsonl(out / "processed" / dataset / f"{split}.jsonl", [{
                "case_id": f"{split}-1", "group_id": "g1", "charges": ["盗窃"],
                "conviction_articles": [], "penalty_type": "fixed_term",
                "imprisonment_months": 6, "factors": {}, "target_defendant": "某甲",
                "fact_raw": "很长的原文" * 100, "fact_conservative": "父事实",
                "fact_strict": "严格事实",
            }])
        (out / "processed" / dataset / "metadata.json").write_text("{}")
    for dataset in ("cail", "cmdl"):
        _jsonl(out / "counterfactuals/qwen38/views/main" / f"{dataset}.jsonl", [{
            "generation_id": f"x{index}", "parent_case_id": "train-1",
            "intervention_type": "invariant", "raw_response": "{}",
            "validation": {"valid": True, "errors": []},
        } for index in range(5)])
        _jsonl(out / "counterfactuals/qwen38" / f"{dataset}.jsonl", [{"cache": True}])
        run = out / "runs/qwen38" / dataset / "M/seed-42"
        _jsonl(run / "predictions/static.jsonl", [{"case_id": "test-1"}])
        (run / "results").mkdir(parents=True)
        (run / "results/static.json").write_text("{}")
        _jsonl(run / "training/train.jsonl", [{"step": 1, "total": 1.0}])
        checkpoint = run / "training/checkpoint-step-00000100"
        checkpoint.mkdir(parents=True)
        (checkpoint / "training_progress.json").write_text('{"steps": 100}')
        (checkpoint / "model.safetensors").write_bytes(b"0" * 4096)
        (run / "training/tensorboard").mkdir()
        (run / "training/tensorboard/events.out").write_bytes(b"0" * 4096)
    (out / "logs").mkdir()
    (out / "logs/vllm.log").write_text("".join(f"line {index}\n" for index in range(5000)))
    (out / "run-summary.json").write_text('{"status": "completed"}')
    return out


def test_bundle_collects_analysis_files_and_skips_heavy_artifacts(tmp_path) -> None:
    out = _output_root(tmp_path)
    console = tmp_path / "main-run.log"
    console.write_text("[phase 1/9] environment\n")
    before = sorted(str(path) for path in out.rglob("*"))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-root", str(out), "--dest", str(tmp_path / "dest"),
         "--console-log", str(console), "--sample", "2"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert sorted(str(path) for path in out.rglob("*")) == before
    [archive] = list((tmp_path / "dest").glob("analysis-bundle-*.tar.gz"))
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        root = names[0].split("/")[0]

        def read(member: str) -> bytes:
            return bundle.extractfile(f"{root}/{member}").read()

        cail_test = [json.loads(line) for line in gzip.decompress(
            read("processed/cail_small-test-labels.jsonl.gz")).decode().splitlines()]
        sample = gzip.decompress(read("counterfactuals/cf-cail-sample.jsonl.gz")).decode()
        slim = gzip.decompress(read("counterfactuals/cf-cail-all-slim.jsonl.gz")).decode()
        manifest = json.loads(read("bundle-manifest.json"))
        vllm_tail = read("logs/vllm-tail.log").decode().splitlines()
    assert f"{root}/runs/qwen38/cail/M/seed-42/predictions/static.jsonl" in names
    assert f"{root}/runs/qwen38/cmdl/M/seed-42/results/static.json" in names
    assert f"{root}/runs/qwen38/cail/M/seed-42/training/train.jsonl" in names
    assert (
        f"{root}/runs/qwen38/cail/M/seed-42/training/checkpoint-step-00000100/"
        "training_progress.json"
    ) in names
    assert not any(name.endswith(".safetensors") or "tensorboard" in name for name in names)
    assert not any(name.endswith("qwen38/cail.jsonl") for name in names)
    assert f"{root}/console/main-run.log" in names
    assert f"{root}/meta/run-summary.json" in names
    assert cail_test == [{
        "case_id": "test-1", "group_id": "g1", "source_path": None, "target_defendant": "某甲",
        "charges": ["盗窃"], "conviction_articles": [], "penalty_type": "fixed_term",
        "imprisonment_months": 6, "factors": {}, "fact_strict": "严格事实",
    }]
    assert len(sample.splitlines()) == 2
    assert json.loads(sample.splitlines()[0])["parent"]["fact_conservative"] == "父事实"
    assert len(slim.splitlines()) == 5
    assert len(vllm_tail) == 3000 and vllm_tail[-1] == "line 4999"
    assert manifest["checkpoints"][0]["steps"] == 100
    assert manifest["missing"]


def test_bundle_reports_missing_output_root(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-root", str(tmp_path / "absent"),
         "--dest", str(tmp_path / "dest")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "OUTPUT_ROOT" in result.stderr
    shutil.rmtree(tmp_path / "dest", ignore_errors=True)
