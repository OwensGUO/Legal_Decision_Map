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


def test_counterfactual_cli_defaults_to_qwen38_config() -> None:
    module = runpy.run_path(str(ROOT / "scripts" / "generate_counterfactuals.py"))
    assert module["parser"]().parse_args([]).config == "configs/cf/qwen38_27b.yaml"


def test_standalone_configs_use_current_server_paths() -> None:
    for name, section, key, expected in (
        ("data/cail_small.yaml", "data", "root", "/data/cguo/datasets/CAIL2018"),
        ("data/cmdl_small.yaml", "data", "root", "/data/cguo/datasets/CMDL"),
        ("model/qwen35_9b_qlora.yaml", "model", "path", "/data/cguo/Qwen3.5-9B"),
    ):
        config = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        assert config[section][key] == expected


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
            "relevant_articles": [264],
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
    assert processed["conviction_articles"] == ["criminal_law:264"]
    assert processed["sentencing_articles"] == []
    assert summary["article_vocabulary"] == ["criminal_law:264"]
    assert "公诉机关指控" not in processed["fact_conservative"]
    assert processed["factors"]["amount"] == 1000.0
    assert (output / "manifest.json").is_file()


def test_build_dataset_keeps_cross_split_group_only_in_held_out_split(tmp_path) -> None:
    source = tmp_path / "source" / "exercise_contest"
    source.mkdir(parents=True)
    duplicate = {
        "fact": "被告人段某实施盗窃并退赃。",
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
    unique_test = {
        **duplicate,
        "fact": "被告人李某实施盗窃并退赃。",
        "meta": {**duplicate["meta"], "criminals": ["李某"]},
    }
    (source / "data_train.json").write_text(
        json.dumps(duplicate, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (source / "data_valid.json").write_text(
        json.dumps(duplicate, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (source / "data_test.json").write_text(
        json.dumps(unique_test, ensure_ascii=False) + "\n", encoding="utf-8"
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
    summary = build_dataset(config_path, output)

    assert (output / "train.jsonl").read_text(encoding="utf-8") == ""
    [valid_record] = [
        json.loads(line)
        for line in (output / "valid.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    [test_record] = [
        json.loads(line)
        for line in (output / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert valid_record["group_id"] != test_record["group_id"]
    assert summary["split_units"] == {"train": 0, "valid": 1, "test": 1}
    assert summary["split_integrity"] == {
        "assignment_policy": "prefer_test_then_valid_then_train",
        "cross_split_groups": 1,
        "dropped_units": {"train": 1, "valid": 0, "test": 0},
    }


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


def test_check_requirements_dry_run_works_without_pythonpath() -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_requirements.py"),
            "--dry-run",
            "--limit",
            "4",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["dry_run"] is True


def test_gpu_probe_reports_pairwise_peer_access() -> None:
    module = runpy.run_path(str(ROOT / "scripts" / "probe_gpu_stack.py"))

    class FakeCuda:
        @staticmethod
        def can_device_access_peer(source: int, target: int) -> bool:
            return {0, 1} == {source, target}

    fake_torch = type("FakeTorch", (), {"cuda": FakeCuda()})()
    assert module["probe_peer_access"](fake_torch, 3) == {
        "pairs": {
            "0->1": True,
            "0->2": False,
            "1->0": True,
            "1->2": False,
            "2->0": False,
            "2->1": False,
        },
        "all_pairs_accessible": False,
    }


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


def test_requirement_version_check_accepts_cuda13_stack_and_rejects_old_stack() -> None:
    module = runpy.run_path(str(ROOT / "scripts" / "check_requirements.py"))
    version_mismatches = module["version_mismatches"]
    expected = {
        "torch": "2.13.0",
        "vllm": "0.30.0",
        "transformers": "5.15.0",
        "accelerate": "1.15.0",
        "peft": "0.21.1",
        "bitsandbytes": "0.50.0",
    }
    assert module["EXPECTED_GPU_VERSIONS"] == expected
    assert version_mismatches({**expected, "torch": "2.13.0+cu130"}, expected) == {}
    assert version_mismatches(
        {**expected, "vllm": "0.19.1"}, expected
    ) == {"vllm": {"expected": "0.30.0", "actual": "0.19.1"}}


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
            "true_articles": [1, 0],
            "predicted_articles": [1, 0],
            "predicted_months": 12.0,
            "true_months": 12.0,
            "penalty_type": "fixed_term",
            "predicted_penalty_type": "fixed_term",
        },
        {
            "group_id": "g2",
            "true_charges": [0, 1],
            "predicted_charges": [0, 1],
            "true_articles": [0, 1],
            "predicted_articles": [0, 1],
            "predicted_months": 18.0,
            "true_months": 18.0,
            "penalty_type": "fixed_term",
            "predicted_penalty_type": "fixed_term",
        },
    ]
    reference = [
        {
            **candidate[0],
            "predicted_charges": [0, 1],
            "predicted_articles": [0, 1],
            "predicted_months": 20.0,
        },
        {
            **candidate[1],
            "predicted_charges": [1, 0],
            "predicted_articles": [1, 0],
            "predicted_months": 26.0,
        },
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
    required = {
        f"{task}_{metric}"
        for task in ("charge", "article", "sentence")
        for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    }
    assert required <= payload.keys()
    assert payload["confidence_intervals"]["charge_macro_f1"]["iterations"] == 20
    assert len(payload["comparison"]["holm_adjusted_p_values"]) == 5
