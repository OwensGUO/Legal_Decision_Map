from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "run.sh"


def test_run_script_has_valid_shell_syntax_and_help() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(RUN_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    help_result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--help"], capture_output=True, text=True, check=False
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "MODE=main|smoke|matrix" in help_result.stdout


def test_run_script_dry_run_lists_gpu_phases_in_safe_order() -> None:
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "MODE": "smoke"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout
    phases = [
        "[phase] environment",
        "[phase] build datasets",
        "[phase] start vLLM",
        "[phase] generate counterfactuals",
        "[phase] stop vLLM",
        "[phase] train and export predictions",
        "[phase] evaluate",
    ]
    positions = [output.index(item) for item in phases]
    assert positions == sorted(positions)
    assert "torch==2.10.0" in output
    assert "vllm serve" in output
    assert "--language-model-only" in output
    assert "--disable-custom-all-reduce" in output
    assert "--enforce-eager" in output
    assert "NCCL_P2P_DISABLE=1" in output
    assert "--resume" in output
    assert "scripts/probe_gpu_stack.py" in output
    assert output.index("scripts/audit_data.py") < output.index("scripts/build_dataset.py")
    assert output.index("validate\\ vLLM\\ JSON") < output.index("generate_counterfactuals.py")


def test_optional_fla_install_is_explicit() -> None:
    base_environment = {**os.environ, "MODE": "smoke", "INSTALL_DEPS": "1"}
    base = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env=base_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert base.returncode == 0, base.stderr
    assert "requirements-optional.txt" not in base.stdout

    optional = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={**base_environment, "INSTALL_FLA": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert optional.returncode == 0, optional.stderr
    assert "requirements-optional.txt" in optional.stdout


def test_run_script_requires_activated_conda_for_real_execution() -> None:
    environment = dict(os.environ)
    environment.pop("CONDA_PREFIX", None)
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Conda" in result.stderr


def test_run_script_rejects_duplicate_or_mismatched_gpu_ids() -> None:
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "GPU_IDS": "0,0,1,2", "NUM_PROCESSES": "4"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unique" in result.stderr


def test_matrix_dry_run_expands_to_78_training_runs() -> None:
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "MODE": "matrix"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    training_commands = [
        line for line in result.stdout.splitlines() if "scripts/train_model.py" in line
    ]
    assert len(training_commands) == 78


def test_main_dry_run_trains_b3_and_pairs_m_evaluation_against_it() -> None:
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "MODE": "main"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    training_commands = [
        line for line in result.stdout.splitlines() if "scripts/train_model.py" in line
    ]
    assert len(training_commands) == 4
    assert sum("--reference-input" in line for line in result.stdout.splitlines()) == 5


def test_failed_real_vllm_probe_aborts_before_generation(tmp_path) -> None:
    conda_prefix = tmp_path / "conda"
    binary_dir = conda_prefix / "bin"
    binary_dir.mkdir(parents=True)
    fake_python = binary_dir / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *json.dumps*) printf '{}' ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_curl = binary_dir / "curl"
    fake_curl.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  */v1/chat/completions*) exit 22 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    cail = tmp_path / "cail"
    cmdl = tmp_path / "cmdl"
    qwen35 = tmp_path / "qwen35"
    qwen36 = tmp_path / "qwen36"
    for path in (cail, cmdl, qwen35, qwen36):
        path.mkdir()
    (qwen35 / "config.json").write_text("{}", encoding="utf-8")
    (qwen36 / "config.json").write_text("{}", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{binary_dir}:{os.environ['PATH']}",
            "CONDA_PREFIX": str(conda_prefix),
            "MODE": "smoke",
            "INFER_MANAGED": "0",
            "CAIL_ROOT": str(cail),
            "CMDL_ROOT": str(cmdl),
            "QWEN35_PATH": str(qwen35),
            "QWEN36_PATH": str(qwen36),
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "vLLM real generation probe failed" in result.stderr
    assert "[phase] generate counterfactuals" not in result.stdout


def test_dry_run_selects_latest_available_training_checkpoint() -> None:
    with TemporaryDirectory() as directory:
        output = Path(directory)
        for dataset in ("cail", "cmdl"):
            training = output / "runs" / dataset / "M" / "seed-42" / "training"
            older = training / "checkpoint-step-00000010"
            newer = training / "checkpoint-step-00000020"
            older.mkdir(parents=True)
            newer.mkdir()
            (older / "training_progress.json").write_text("{}", encoding="utf-8")
            (newer / "training_progress.json").write_text("{}", encoding="utf-8")
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))
        result = subprocess.run(
            ["bash", str(RUN_SCRIPT), "--dry-run"],
            cwd=ROOT,
            env={**os.environ, "MODE": "smoke", "OUTPUT_ROOT": str(output)},
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("--resume-from-checkpoint") == 2
    assert result.stdout.count("checkpoint-step-00000020") == 2
