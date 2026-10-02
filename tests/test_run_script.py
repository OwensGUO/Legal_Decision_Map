from __future__ import annotations

import errno
import json
import os
import re
import select
import shlex
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "run.sh"


def test_readme_documents_progress_controls() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for value in ("PROGRESS=auto", "PROGRESS=always", "PROGRESS=never", "NO_COLOR"):
        assert value in readme
    assert "stderr" in readme
    assert "stdout" in readme


@pytest.mark.parametrize("generator", ["qwen38", "qwen36"])
def test_provenance_preflight_guards_both_generator_namespaces_before_server(tmp_path, generator):
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"], cwd=ROOT,
        env={**os.environ, "MODE": "smoke", "GENERATOR_MODEL": generator,
             "OUTPUT_ROOT": str(tmp_path), f"{generator.upper()}_PATH": "/custom/checkpoint"},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line[2:]) for line in result.stdout.splitlines()
                if line.startswith("$ ")]
    preflight = [args for args in commands if str(ROOT / "scripts/check_generator_provenance.py")
                 in args]
    assert preflight
    args = preflight[0]
    assert "generator.checkpoint_path=/custom/checkpoint" in args
    assert f"generator.selector={generator}" in args
    assert str(tmp_path / "runs" / generator) in args
    assert str(tmp_path / "counterfactuals" / generator) in args
    assert result.stdout.index("check_generator_provenance.py") < result.stdout.index("vllm serve")
    generation = [args for args in commands if str(ROOT / "scripts/generate_counterfactuals.py")
                  in args]
    for args in generation:
        assert "generator.checkpoint_path=/custom/checkpoint" in args
        assert f"generator.selector={generator}" in args
        assert str(tmp_path / "runs" / generator) in args


@pytest.mark.parametrize("generator", ["qwen38", "qwen36"])
@pytest.mark.parametrize("artifact_root", ["counterfactuals", "runs"])
@pytest.mark.parametrize("provenance_state", ["missing", "changed_checkpoint"])
def test_real_pipeline_rejects_unprovenanced_artifacts_before_generation_or_training(
    tmp_path, generator, artifact_root, provenance_state,
):
    binary_dir = tmp_path / "conda/bin"
    binary_dir.mkdir(parents=True)
    wrapper = binary_dir / "python"
    wrapper.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        "if sys.argv[1].endswith('check_generator_provenance.py'):\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    wrapper.chmod(0o755)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{}')
    (checkpoint / "model.safetensors").write_bytes(b"fake weights")
    if provenance_state == "changed_checkpoint":
        from legal_landscape.counterfactual.provenance import ensure_manifest, generator_identity

        identity = generator_identity({
            "selector": generator, "model_path": {"qwen38": "Qwen3.8-27B",
                                                   "qwen36": "Qwen3.6-27B"}[generator],
            "checkpoint_path": str(checkpoint), "model_revision": "local",
        })
        manifest = tmp_path / "outputs/counterfactuals" / generator / "generator-provenance.json"
        ensure_manifest(manifest, identity, artifact_roots=[manifest.parent])
        (checkpoint / "config.json").write_text('{"new_checkpoint":true}')
    artifact = tmp_path / "outputs" / artifact_root / generator / "stale-artifact"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("existing output")
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT)], cwd=ROOT,
        env={**os.environ, "MODE": "smoke", "GENERATOR_MODEL": generator,
             "OUTPUT_ROOT": str(tmp_path / "outputs"), "CONDA_PREFIX": str(binary_dir.parent),
             "PATH": f"{binary_dir}:{os.environ['PATH']}", "INFER_MANAGED": "0",
             "CAIL_ROOT": str(tmp_path), "CMDL_ROOT": str(tmp_path),
             "QWEN35_PATH": str(checkpoint), f"{generator.upper()}_PATH": str(checkpoint)},
        capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode != 0
    expected = ("Missing generator provenance" if provenance_state == "missing"
                else "Generator provenance mismatch")
    assert expected in result.stderr
    assert "new OUTPUT_ROOT" in result.stderr
    assert "[phase] start vLLM" not in result.stdout
    assert "[phase] generate counterfactuals" not in result.stdout
    assert "[phase] train and export predictions" not in result.stdout


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


def test_dry_run_numbers_all_phases_without_terminal_control() -> None:
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "MODE": "smoke", "PROGRESS": "always"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for index, name in enumerate(
        (
            "environment", "validate generator provenance", "audit data",
            "build datasets", "start vLLM", "generate counterfactuals",
            "stop vLLM", "train and export predictions", "evaluate",
        ),
        start=1,
    ):
        assert f"[phase {index}/9] {name}" in result.stdout
        assert re.search(
            rf"\[phase {index}/9\] {re.escape(name)} completed in \d+s",
            result.stdout,
        )
    assert re.search(r"\[done\] Pipeline completed in \d+s\. Outputs: ", result.stdout)
    assert "\x1b[" not in result.stdout + result.stderr
    assert "\r" not in result.stdout + result.stderr


@pytest.mark.parametrize("progress", ["sometimes", ""])
def test_invalid_progress_mode_aborts_before_any_phase(progress) -> None:
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "PROGRESS": progress},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "PROGRESS must be auto, always, or never" in result.stderr
    assert "[phase" not in result.stdout


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
        "[phase 1/9] environment",
        "[phase 4/9] build datasets",
        "[phase 5/9] start vLLM",
        "[phase 6/9] generate counterfactuals",
        "[phase 7/9] stop vLLM",
        "[phase 8/9] train and export predictions",
        "[phase 9/9] evaluate",
    ]
    positions = [output.index(item) for item in phases]
    assert positions == sorted(positions)
    assert "torch==2.13.0" in output
    assert "vllm==0.30.0" in output
    assert "CUDA 13.0" in output
    assert "/data/cguo/Qwen3.5-9B" in output
    assert "/data/cguo/Qwen3.8-27B" in output
    assert "/data/cguo/datasets/CAIL2018" in output
    assert "/data/cguo/datasets/CMDL" in output
    assert "configs/cf/qwen38_27b.yaml" in output
    assert "counterfactuals/qwen38" in output
    assert "vllm serve" in output
    assert "--language-model-only" in output
    assert "--disable-custom-all-reduce" in output
    assert "--enable-prefix-caching" in output
    assert "--enforce-eager" not in output
    assert "NCCL_P2P_DISABLE=1" in output
    assert "--resume" in output
    assert "scripts/probe_gpu_stack.py" in output
    assert "--output" in output
    assert "gpu-probe.json" in output
    assert "CUDA_VISIBLE_DEVICES=4\\,5\\,6\\,7" in output
    assert "--num_processes 4" in output
    assert output.index("scripts/audit_data.py") < output.index("scripts/build_dataset.py")
    assert output.index("validate\\ vLLM\\ JSON") < output.index("generate_counterfactuals.py")


def test_dry_run_selects_qwen36_without_reusing_qwen38_counterfactuals() -> None:
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "MODE": "smoke", "GENERATOR_MODEL": "qwen36"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "/data/cguo/Qwen3.6-27B" in result.stdout
    assert "configs/cf/qwen36_27b.yaml" in result.stdout
    assert "counterfactuals/qwen36" in result.stdout
    assert "counterfactuals/qwen38" not in result.stdout
    assert "--served-model-name Qwen3.6-27B" in result.stdout


@pytest.mark.parametrize(
    "setting", [{"GENERATOR_MODEL": "unknown"}, {"P2P_POLICY": "unknown"}]
)
def test_invalid_generator_or_p2p_policy_aborts_before_any_phase(setting) -> None:
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, **setting},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "[phase]" not in result.stdout


@pytest.mark.parametrize("policy,disabled", [("enable", False), ("disable", True)])
def test_dry_run_respects_explicit_p2p_policy(policy, disabled) -> None:
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "MODE": "smoke", "P2P_POLICY": policy, "VLLM_ENFORCE_EAGER": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    command = next(line for line in result.stdout.splitlines() if "vllm serve" in line)
    assert ("NCCL_P2P_DISABLE=1" in command) is disabled
    assert ("--disable-custom-all-reduce" in command) is disabled
    assert "--enforce-eager" in command


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
    assert "/data/cguo/models/RoBERTa" in result.stdout
    assert "/data/cguo/models/Lawformer" in result.stdout


def test_baseline_model_paths_can_be_overridden() -> None:
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={
            **os.environ,
            "MODE": "matrix",
            "EXPERIMENTS": "B1 B2",
            "SEEDS": "42",
            "ROBERTA_PATH": "/custom/RoBERTa",
            "LAWFORMER_PATH": "/custom/Lawformer",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    training_commands = [
        shlex.split(line[2:])
        for line in result.stdout.splitlines()
        if line.startswith("$ ") and "scripts/train_model.py" in line
    ]
    assert len(training_commands) == 4
    for args in training_commands:
        experiment = args[args.index("--experiment") + 1]
        expected = "/custom/RoBERTa" if experiment == "B1" else "/custom/Lawformer"
        assert f"model.path={expected}" in args


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


@pytest.mark.parametrize("generator", ["qwen38", "qwen36"])
def test_generator_namespaces_training_predictions_evaluation_and_b3_references(
    tmp_path, generator
) -> None:
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={
            **os.environ,
            "MODE": "main",
            "GENERATOR_MODEL": generator,
            "OUTPUT_ROOT": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    commands = [
        shlex.split(line[2:])
        for line in result.stdout.splitlines()
        if line.startswith("$ ")
    ]
    training = [args for args in commands if str(ROOT / "scripts/train_model.py") in args]
    evaluations = [args for args in commands if str(ROOT / "scripts/evaluate_model.py") in args]
    assert len(training) == 4
    assert len(evaluations) == 10
    for args in training:
        dataset = Path(args[args.index("--train-data") + 1]).parent.name.removesuffix("_small")
        experiment = args[args.index("--experiment") + 1]
        run_root = tmp_path / "runs" / generator / dataset / experiment / "seed-42"
        assert args[args.index("--output-dir") + 1] == str(run_root / "training")
        assert args[args.index("--prediction-dir") + 1] == str(run_root / "predictions")
        assert args[args.index("--train-data") + 1] == str(
            tmp_path / "processed" / f"{dataset}_small" / "train.jsonl"
        )
    references = []
    for args in evaluations:
        input_path = Path(args[args.index("--input") + 1])
        output_path = Path(args[args.index("--output") + 1])
        assert input_path.is_relative_to(tmp_path / "runs" / generator)
        assert input_path.parent.name == "predictions"
        assert output_path.parent == input_path.parent.parent / "results"
        if "--reference-input" in args:
            reference = Path(args[args.index("--reference-input") + 1])
            dataset = input_path.relative_to(tmp_path / "runs" / generator).parts[0]
            assert reference == (
                tmp_path / "runs" / generator / dataset / "B3" / "seed-42"
                / "predictions" / input_path.name
            )
            references.append(reference)
    assert len(references) == 5


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
    qwen38 = tmp_path / "qwen38"
    for path in (cail, cmdl, qwen35, qwen38):
        path.mkdir()
    (qwen35 / "config.json").write_text("{}", encoding="utf-8")
    (qwen38 / "config.json").write_text("{}", encoding="utf-8")
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
            "QWEN38_PATH": str(qwen38),
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "vLLM real generation probe failed" in result.stderr
    assert "[phase] generate counterfactuals" not in result.stdout


@pytest.mark.parametrize("progress", ["never", "always"])
@pytest.mark.parametrize(
    "policy,probe_json,disabled",
    [
        ("auto", '{"peer_access":{"all_pairs_accessible":true}}', False),
        ("auto", '{"peer_access":{"all_pairs_accessible":false}}', True),
        ("auto", '{"peer_access":{"all_pairs_accessible":"true"}}', True),
        ("auto", '{"peer_access":{"all_pairs_accessible":1}}', True),
        ("auto", '{"peer_access":null}', True),
        ("auto", '{}', True),
        ("auto", 'invalid JSON', True),
        ("auto", 'missing', True),
        ("enable", '{"peer_access":{"all_pairs_accessible":false}}', False),
        ("disable", '{"peer_access":{"all_pairs_accessible":true}}', True),
    ],
)
def test_real_server_p2p_flags_follow_policy_and_boolean_probe(
    tmp_path, policy, probe_json, disabled, progress
) -> None:
    """Keep orchestration real while replacing GPU/server boundary commands."""
    conda_prefix = tmp_path / "conda"
    binary_dir = conda_prefix / "bin"
    binary_dir.mkdir(parents=True)
    launch_args = tmp_path / "server-args.json"
    fake_python = binary_dir / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if args[0].endswith('probe_gpu_stack.py'):\n"
        "    payload = os.environ['TEST_PROBE_JSON']\n"
        "    if payload != 'missing':\n"
        "        pathlib.Path(args[args.index('--output') + 1]).write_text(payload)\n"
        "elif args[0] == '-c' and not args[1].startswith(\n"
        "    ('import sys; assert', 'import accelerate')\n"
        "):\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, *args])\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_setsid = binary_dir / "setsid"
    fake_setsid.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "os.execvp(sys.argv[1], sys.argv[1:])\n",
        encoding="utf-8",
    )
    fake_setsid.chmod(0o755)
    fake_vllm = binary_dir / "vllm"
    fake_vllm.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "payload = {'arguments': sys.argv[1:], 'p2p_disable': os.getenv('NCCL_P2P_DISABLE'), "
        "'gpu_ids': os.getenv('CUDA_VISIBLE_DEVICES')}\n"
        "pathlib.Path(os.environ['TEST_LAUNCH_ARGS']).write_text(json.dumps(payload))\n",
        encoding="utf-8",
    )
    fake_vllm.chmod(0o755)
    fake_curl = binary_dir / "curl"
    fake_curl.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  */v1/chat/completions*) exit 22 ;;\n"
        "  *) test -f \"$TEST_LAUNCH_ARGS\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    (binary_dir / "sleep").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (binary_dir / "sleep").chmod(0o755)
    paths = {name: tmp_path / name for name in ("cail", "cmdl", "qwen35", "qwen38")}
    for path in paths.values():
        path.mkdir()
    for name in ("qwen35", "qwen38"):
        (paths[name] / "config.json").write_text("{}", encoding="utf-8")
    output_root = tmp_path / "outputs"
    result = subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{binary_dir}:{os.environ['PATH']}",
            "CONDA_PREFIX": str(conda_prefix),
            "MODE": "smoke",
            "P2P_POLICY": policy,
            "PROGRESS": progress,
            "NCCL_P2P_DISABLE": "1",
            "TEST_PROBE_JSON": probe_json,
            "TEST_LAUNCH_ARGS": str(launch_args),
            "CAIL_ROOT": str(paths["cail"]),
            "CMDL_ROOT": str(paths["cmdl"]),
            "QWEN35_PATH": str(paths["qwen35"]),
            "QWEN38_PATH": str(paths["qwen38"]),
            "OUTPUT_ROOT": str(output_root),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode != 0
    assert "vLLM real generation probe failed" in result.stderr
    if progress == "never":
        assert re.search(r"(?:^|\n)Waiting for vLLM \(0s/\d+s\)\n", result.stderr)
        assert "\r" not in result.stderr
    else:
        assert "Waiting for vLLM" in result.stderr
        assert "\x1b[2K" in result.stderr
    assert len(re.findall(r"\[phase 5/9\] start vLLM failed in \d+s", result.stderr)) == 1
    assert "[phase] generate counterfactuals" not in result.stdout
    launch = json.loads(launch_args.read_text(encoding="utf-8"))
    arguments = launch["arguments"]
    assert launch["p2p_disable"] == ("1" if disabled else None)
    assert ("--disable-custom-all-reduce" in arguments) is disabled
    assert "--enforce-eager" not in arguments
    assert "--enable-prefix-caching" in arguments
    assert launch["gpu_ids"] == "4,5,6,7"
    assert arguments[arguments.index("--tensor-parallel-size") + 1] == "4"
    assert arguments[arguments.index("--host") + 1] == "127.0.0.1"
    if probe_json != "missing":
        assert (output_root / "gpu-probe.json").read_text() == probe_json
    summary = json.loads((output_root / "run-summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["generator_model"] == "qwen38"
    assert summary["generator_path"] == str(paths["qwen38"])
    assert summary["p2p_policy"] == policy


def _run_fake_vllm_wait(
    tmp_path, *, health_mode, progress, columns=None, no_color=False, stderr_tty=False
):
    binary_dir = tmp_path / "conda/bin"
    binary_dir.mkdir(parents=True)
    launch_marker = tmp_path / "vllm-launched"
    fake_python = binary_dir / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "if args[0] == '-c' and not args[1].startswith("
        "('import sys; assert', 'import accelerate')):\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, *args])\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_setsid = binary_dir / "setsid"
    fake_setsid.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        "os.execvp(sys.argv[1], sys.argv[1:])\n",
        encoding="utf-8",
    )
    fake_setsid.chmod(0o755)
    fake_vllm = binary_dir / "vllm"
    fake_vllm.write_text(
        f"#!{sys.executable}\nimport os, pathlib, time\n"
        "pathlib.Path(os.environ['TEST_LAUNCH_MARKER']).touch()\n"
        "if os.environ['TEST_HEALTH_MODE'] == 'timeout':\n"
        "    time.sleep(2)\n",
        encoding="utf-8",
    )
    fake_vllm.chmod(0o755)
    fake_curl = binary_dir / "curl"
    fake_curl.write_text(
        "#!/bin/sh\n"
        "test -f \"$TEST_LAUNCH_MARKER\" || exit 22\n"
        "case \"$TEST_HEALTH_MODE\" in\n"
        "  ready) case \"$*\" in *v1/chat/completions*) exit 22 ;; *) exit 0 ;; esac ;;\n"
        "  death) /bin/sleep 0.1; exit 22 ;;\n"
        "  timeout) exit 22 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    fake_sleep = binary_dir / "sleep"
    fake_sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_sleep.chmod(0o755)
    paths = {name: tmp_path / name for name in ("cail", "cmdl", "qwen35", "qwen38")}
    for path in paths.values():
        path.mkdir()
    for name in ("qwen35", "qwen38"):
        (paths[name] / "config.json").write_text("{}", encoding="utf-8")
    environment = {
        **os.environ,
        "PATH": f"{binary_dir}:{os.environ['PATH']}",
        "CONDA_PREFIX": str(binary_dir.parent),
        "MODE": "smoke",
        "P2P_POLICY": "disable",
        "PROGRESS": progress,
        "TEST_HEALTH_MODE": health_mode,
        "TEST_LAUNCH_MARKER": str(launch_marker),
        "INFER_TIMEOUT": "0" if health_mode == "timeout" else "30",
        "CAIL_ROOT": str(paths["cail"]),
        "CMDL_ROOT": str(paths["cmdl"]),
        "QWEN35_PATH": str(paths["qwen35"]),
        "QWEN38_PATH": str(paths["qwen38"]),
        "OUTPUT_ROOT": str(tmp_path / "outputs"),
    }
    if columns is not None:
        environment["COLUMNS"] = str(columns)
    if no_color:
        environment["NO_COLOR"] = "1"
    if stderr_tty:
        master_fd, slave_fd = os.openpty()
        try:
            process = subprocess.Popen(
                ["bash", str(RUN_SCRIPT)], cwd=ROOT, env=environment,
                stdout=subprocess.DEVNULL, stderr=slave_fd,
            )
            os.close(slave_fd)
            chunks = []
            deadline = time.monotonic() + 15
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([master_fd], [], [], remaining)[0]:
                    process.kill()
                    process.wait()
                    raise subprocess.TimeoutExpired(process.args, 15)
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError as error:
                    if error.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break
                chunks.append(chunk)
            stderr = b"".join(chunks)
            returncode = process.wait(timeout=max(0, deadline - time.monotonic()))
        finally:
            os.close(master_fd)
    else:
        result = subprocess.run(
            ["bash", str(RUN_SCRIPT)], cwd=ROOT, env=environment,
            capture_output=True, check=False, timeout=15,
        )
        stderr = result.stderr
        returncode = result.returncode
    return stderr.decode("utf-8"), returncode


@pytest.mark.parametrize(
    "health_mode,diagnostic",
    [
        ("timeout", "Timed out waiting for vLLM"),
        ("death", "vLLM exited before becoming healthy"),
    ],
)
def test_dynamic_wait_clears_before_failure_diagnostic(
    tmp_path, health_mode, diagnostic
) -> None:
    stderr, returncode = _run_fake_vllm_wait(
        tmp_path, health_mode=health_mode, progress="always"
    )
    assert returncode == 1
    assert "\r\x1b[2K" in stderr
    assert re.search(rf"vLLM wait failed after \d+s\n{diagnostic}", stderr)
    assert stderr.count("vLLM wait failed after") == 1


@pytest.mark.parametrize("columns", [18, 8, 4])
def test_narrow_wait_keeps_each_status_within_terminal_width(tmp_path, columns) -> None:
    stderr, returncode = _run_fake_vllm_wait(
        tmp_path, health_mode="ready", progress="always", columns=columns
    )
    assert returncode == 1
    assert "vLLM real generation probe failed" in stderr
    wait_section = stderr.split("[phase 5/9] start vLLM\n", 1)[1].split(
        "vLLM real generation probe failed", 1
    )[0]
    wait_lines = [
        line.replace("\x1b[2K", "")
        for line in re.split(r"[\r\n]", wait_section)
        if line.strip("\x1b[2K")
    ]
    assert wait_lines
    assert all(len(line) < columns for line in wait_lines)
    if columns <= 8:
        assert "\x1b[2K" not in stderr


def test_auto_with_redirected_stderr_stays_plain(tmp_path) -> None:
    stderr, returncode = _run_fake_vllm_wait(
        tmp_path, health_mode="ready", progress="auto"
    )
    assert returncode == 1
    assert "Waiting for vLLM (0s/30s)\n" in stderr
    assert "\x1b[" not in stderr


def test_auto_with_tty_stderr_uses_dynamic_wait(tmp_path) -> None:
    stderr, returncode = _run_fake_vllm_wait(
        tmp_path, health_mode="ready", progress="auto", stderr_tty=True
    )
    assert returncode == 1
    assert "Waiting for vLLM" in stderr
    assert "\r\x1b[2K" in stderr


@pytest.mark.parametrize("progress", ["never", "auto"])
@pytest.mark.parametrize("columns", [4, 8])
def test_plain_wait_keeps_full_records_despite_narrow_columns(
    tmp_path, columns, progress
) -> None:
    stderr, returncode = _run_fake_vllm_wait(
        tmp_path, health_mode="ready", progress=progress, columns=columns
    )
    assert returncode == 1
    assert "Waiting for vLLM (0s/30s)\n" in stderr
    assert re.search(r"vLLM wait ready after \d+s\n", stderr)
    assert "\x1b[" not in stderr


def test_no_color_preserves_forced_wait_progress(tmp_path) -> None:
    stderr, returncode = _run_fake_vllm_wait(
        tmp_path, health_mode="ready", progress="always", no_color=True
    )
    assert returncode == 1
    assert "\r\x1b[2K" in stderr
    assert "Waiting for vLLM" in stderr
    assert re.search(r"\x1b\[[0-9;]*m", stderr) is None


@pytest.mark.parametrize("generator", ["qwen38", "qwen36"])
def test_dry_run_selects_latest_available_training_checkpoint(generator) -> None:
    with TemporaryDirectory() as directory:
        output = Path(directory)
        for dataset in ("cail", "cmdl"):
            training = output / "runs" / generator / dataset / "M" / "seed-42" / "training"
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
            env={
                **os.environ,
                "MODE": "smoke",
                "GENERATOR_MODEL": generator,
                "OUTPUT_ROOT": str(output),
            },
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("--resume-from-checkpoint") == 2
    assert result.stdout.count("checkpoint-step-00000020") == 2


def test_switching_generator_ignores_other_generator_and_legacy_checkpoints(tmp_path) -> None:
    for prefix in (tmp_path / "runs" / "qwen38", tmp_path / "runs"):
        for dataset in ("cail", "cmdl"):
            checkpoint = prefix / dataset / "M" / "seed-42" / "training" / "checkpoint-final"
            checkpoint.mkdir(parents=True)
            (checkpoint / "training_progress.json").write_text("{}", encoding="utf-8")
    for generator, expected_resume_count in (("qwen38", 2), ("qwen36", 0)):
        result = subprocess.run(
            ["bash", str(RUN_SCRIPT), "--dry-run"],
            cwd=ROOT,
            env={
                **os.environ,
                "MODE": "smoke",
                "GENERATOR_MODEL": generator,
                "OUTPUT_ROOT": str(tmp_path),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.count("--resume-from-checkpoint") == expected_resume_count
        if generator == "qwen38":
            checkpoint = tmp_path / "runs/qwen38/cail/M/seed-42/training/checkpoint-final"
            assert str(checkpoint) in result.stdout
        else:
            assert f"{tmp_path}/runs/qwen38" not in result.stdout
