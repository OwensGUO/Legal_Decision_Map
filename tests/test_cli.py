from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import yaml

from legal_landscape.data.build import build_dataset

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "audit_environment.py",
    "audit_data.py",
    "build_dataset.py",
    "generate_counterfactuals.py",
    "train_model.py",
    "evaluate_model.py",
    "check_requirements.py",
    "probe_gpu_stack.py",
)


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_all_cli_help_paths() -> None:
    for name in SCRIPTS:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name), "--help"],
            cwd=ROOT,
            env=_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"
        assert "usage:" in result.stdout.lower()


def test_heavy_clis_default_to_dry_run() -> None:
    for name, extra in (
        ("generate_counterfactuals.py", []),
        ("train_model.py", []),
        ("evaluate_model.py", []),
    ):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name), *extra],
            cwd=ROOT,
            env=_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["dry_run"] is True


def test_build_dataset_writes_traceable_processed_records(tmp_path) -> None:
    source = tmp_path / "source" / "exercise_contest"
    source.mkdir(parents=True)
    row = {
        "fact": "被告人段某涉案金额1000元，已退赃。公诉机关指控其犯盗窃罪。",
        "meta": {
            "criminals": ["段某"],
            "accusation": ["盗窃"],
            "term_of_imprisonment": {
                "death_penalty": False,
                "life_imprisonment": False,
                "imprisonment": 12,
            },
        },
    }
    for name in ("data_train.json", "data_valid.json", "data_test.json"):
        split_row = {**row, "fact": f"{row['fact']}{name}"}
        (source / name).write_text(
            json.dumps(split_row, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "dataset": "cail",
                    "root": str(tmp_path / "source"),
                    "variant": "exercise_contest",
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "processed"
    summary = build_dataset(config_path, output, limit=1)
    processed = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    assert summary["split_units"] == {"train": 1, "valid": 1, "test": 1}
    assert processed["case_id"] == processed["group_id"]
    assert processed["source_path"].endswith("data_train.json:1")
    assert processed["fact_raw"] == f"{row['fact']}data_train.json"
    assert "公诉机关指控" not in processed["fact_conservative"]
    assert processed["factors"]["amount"] == 1000.0
    assert (output / "manifest.json").is_file()


def test_environment_and_requirement_dry_runs_are_read_only() -> None:
    for name in ("audit_environment.py", "check_requirements.py", "probe_gpu_stack.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name), "--dry-run", "--limit", "1"],
            cwd=ROOT,
            env=_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["dry_run"] is True


def test_requirement_failures_include_every_requested_dependency_group() -> None:
    module = runpy.run_path(str(ROOT / "scripts" / "check_requirements.py"))
    has_requirement_failures = module["has_requirement_failures"]
    packages = {
        "yaml": "6.0.2",
        "torch": "2.10.0",
        "vllm": "ERROR: ImportError: missing CUDA extension",
    }
    assert has_requirement_failures(packages, ("yaml", "torch", "vllm"))
    assert not has_requirement_failures(packages, ("yaml", "torch"))


def test_requirement_version_check_accepts_local_suffix_and_rejects_drift() -> None:
    module = runpy.run_path(str(ROOT / "scripts" / "check_requirements.py"))
    version_mismatches = module["version_mismatches"]
    expected = {"torch": "2.10.0", "vllm": "0.19.1"}
    assert version_mismatches(
        {"torch": "2.10.0+cu128", "vllm": "0.19.1"}, expected
    ) == {}
    assert version_mismatches(
        {"torch": "2.5.1+cu121", "vllm": "0.19.1"}, expected
    ) == {"torch": {"expected": "2.10.0", "actual": "2.5.1+cu121"}}


def test_environment_audit_reports_physical_cuda_visibility(monkeypatch) -> None:
    module = runpy.run_path(str(ROOT / "scripts" / "audit_environment.py"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    payload = module["audit"](0)
    assert payload["cuda_visible_devices"] == ["4", "5", "6", "7"]


def test_evaluation_cli_writes_bootstrap_and_five_holm_results(tmp_path) -> None:
    candidate = [
        {
            "group_id": "g1",
            "true_charges": [1, 0],
            "predicted_charges": [1, 0],
            "predicted_months": 12.0,
            "true_months": 12.0,
            "penalty_type": "fixed_term",
        },
        {
            "group_id": "g2",
            "true_charges": [0, 1],
            "predicted_charges": [0, 1],
            "predicted_months": 18.0,
            "true_months": 18.0,
            "penalty_type": "fixed_term",
        },
    ]
    reference = [
        {**candidate[0], "predicted_charges": [0, 1], "predicted_months": 20.0},
        {**candidate[1], "predicted_charges": [1, 0], "predicted_months": 26.0},
    ]
    candidate_path = tmp_path / "candidate.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    output_path = tmp_path / "evaluation.json"
    candidate_path.write_text(
        "\n".join(json.dumps(row) for row in candidate) + "\n", encoding="utf-8"
    )
    reference_path.write_text(
        "\n".join(json.dumps(row) for row in reference) + "\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_model.py"),
            "--kind",
            "static",
            "--input",
            str(candidate_path),
            "--reference-input",
            str(reference_path),
            "--output",
            str(output_path),
            "--bootstrap-iterations",
            "20",
        ],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["confidence_intervals"]["macro_f1"]["iterations"] == 20
    assert len(payload["comparison"]["holm_adjusted_p_values"]) == 5
