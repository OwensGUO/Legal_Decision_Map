from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _checkpoint(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"qwen3"}')
    (path / "model.safetensors").write_bytes(b"fake weights")
    (path / "tokenizer_config.json").write_text('{"eos_token":"<end>"}')
    return path


def _identity(path: Path, selector: str = "qwen38") -> dict:
    provenance = importlib.import_module("legal_landscape.counterfactual.provenance")
    return provenance.generator_identity(
        {
            "selector": selector,
            "model_path": "served-model",
            "checkpoint_path": str(path),
            "model_revision": "local",
            "backend": "vllm",
        }
    )


def test_checkpoint_identity_is_deterministic_and_tracks_metadata_and_weight_inventory(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    first = _identity(checkpoint)
    assert first == _identity(checkpoint / ".")
    assert first["schema_version"] == 1
    assert first["generator_selector"] == "qwen38"
    assert first["served_model_name"] == "served-model"
    assert first["checkpoint_path"] == str(checkpoint.resolve())
    assert first["configured_revision"] == "local"
    assert len(first["checkpoint_fingerprint"]) == 64
    (checkpoint / "README.md").write_text("unrelated notes")
    assert _identity(checkpoint) == first
    (checkpoint / "tokenizer_config.json").write_text('{"eos_token":"<new>"}')
    assert _identity(checkpoint)["checkpoint_fingerprint"] != first["checkpoint_fingerprint"]
    second = _identity(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"larger fake weights")
    assert _identity(checkpoint)["checkpoint_fingerprint"] != second["checkpoint_fingerprint"]


def test_checkpoint_path_distinguishes_identical_local_copies(tmp_path):
    first = _identity(_checkpoint(tmp_path / "original"))
    second = _identity(_checkpoint(tmp_path / "copy"))
    assert first["checkpoint_fingerprint"] == second["checkpoint_fingerprint"]
    assert first["checkpoint_path"] != second["checkpoint_path"]
    provenance = importlib.import_module("legal_landscape.counterfactual.provenance")
    manifest = tmp_path / "outputs/generator-provenance.json"
    provenance.ensure_manifest(manifest, first, artifact_roots=[manifest.parent])
    with pytest.raises(ValueError, match="provenance mismatch"):
        provenance.ensure_manifest(manifest, second, artifact_roots=[manifest.parent])


@pytest.mark.parametrize("missing", ["config.json", "model.safetensors"])
def test_incomplete_checkpoint_fails_clearly(tmp_path, missing):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / missing).unlink()
    with pytest.raises(ValueError, match="checkpoint identity"):
        _identity(checkpoint)


def test_missing_index_shard_fails_clearly(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer":"missing.safetensors"}}'
    )
    with pytest.raises(ValueError, match="checkpoint identity"):
        _identity(checkpoint)


def test_fresh_manifest_resumes_without_rewriting_and_rejects_identity_changes(tmp_path):
    identity = _identity(_checkpoint(tmp_path / "checkpoint"))
    provenance = importlib.import_module("legal_landscape.counterfactual.provenance")
    manifest = tmp_path / "counterfactuals/qwen38/generator-provenance.json"
    roots = [manifest.parent, tmp_path / "runs/qwen38"]
    provenance.ensure_manifest(manifest, identity, artifact_roots=roots)
    original = manifest.read_bytes()
    modified = manifest.stat().st_mtime_ns
    assert json.loads(original) == identity
    (manifest.parent / "cail.jsonl").write_text("generated artifact")
    provenance.ensure_manifest(manifest, identity, artifact_roots=roots)
    assert manifest.read_bytes() == original
    assert manifest.stat().st_mtime_ns == modified
    for key in ("checkpoint_path", "checkpoint_fingerprint", "configured_revision"):
        with pytest.raises(ValueError, match="new OUTPUT_ROOT.*migration/removal"):
            provenance.ensure_manifest(manifest, {**identity, key: "changed"}, artifact_roots=roots)
        assert manifest.read_bytes() == original


@pytest.mark.parametrize(
    "artifact",
    ["counterfactuals/qwen38/cail.jsonl", "runs/qwen38/cail/M/training/checkpoint/state.bin"],
)
def test_existing_artifacts_without_manifest_are_rejected(tmp_path, artifact):
    identity = _identity(_checkpoint(tmp_path / "checkpoint"))
    provenance = importlib.import_module("legal_landscape.counterfactual.provenance")
    existing = tmp_path / artifact
    existing.parent.mkdir(parents=True)
    existing.write_text("old artifact")
    manifest = tmp_path / "counterfactuals/qwen38/generator-provenance.json"
    with pytest.raises(ValueError, match="Missing.*new OUTPUT_ROOT"):
        provenance.ensure_manifest(
            manifest,
            identity,
            artifact_roots=[
                manifest.parent,
                tmp_path / "runs/qwen38",
            ],
        )
    assert not manifest.exists()


def test_mock_identity_is_explicit_and_does_not_require_checkpoint(tmp_path):
    provenance = importlib.import_module("legal_landscape.counterfactual.provenance")
    identity = provenance.generator_identity({"selector": "qwen38"}, mock=True)
    assert identity["backend"] == "mock"
    assert identity["served_model_name"] == "mock-v1"
    assert identity["checkpoint_path"] == "builtin:mock-v1"
    assert len(identity["checkpoint_fingerprint"]) == 64


def test_checkpoint_metadata_limit_is_enforced_before_reading_config(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    provenance = importlib.import_module("legal_landscape.counterfactual.provenance")
    monkeypatch.setattr(provenance, "_METADATA_LIMIT", 1)
    original_read_text = Path.read_text

    def guarded_read(path, *args, **kwargs):
        assert path != checkpoint / "config.json", "oversized config must not be read"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    with pytest.raises(ValueError, match="64 MiB bound"):
        _identity(checkpoint)


def test_manifest_publication_failure_leaves_no_partial_manifest(tmp_path, monkeypatch):
    identity = _identity(_checkpoint(tmp_path / "checkpoint"))
    provenance = importlib.import_module("legal_landscape.counterfactual.provenance")
    manifest = tmp_path / "outputs/generator-provenance.json"

    def failed_link(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(provenance.os, "link", failed_link)
    with pytest.raises(OSError, match="publication failure"):
        provenance.ensure_manifest(manifest, identity, artifact_roots=[manifest.parent])
    assert list(manifest.parent.iterdir()) == []


def test_competing_manifest_cannot_be_overwritten(tmp_path, monkeypatch):
    identity = _identity(_checkpoint(tmp_path / "checkpoint"))
    provenance = importlib.import_module("legal_landscape.counterfactual.provenance")
    manifest = tmp_path / "outputs/generator-provenance.json"
    competing = {**identity, "checkpoint_path": "/other/model"}

    def competing_link(source, destination):
        Path(destination).write_text(json.dumps(competing))
        raise FileExistsError("another process published first")

    monkeypatch.setattr(provenance.os, "link", competing_link)
    with pytest.raises(ValueError, match="provenance mismatch"):
        provenance.ensure_manifest(manifest, identity, artifact_roots=[manifest.parent])
    assert json.loads(manifest.read_text()) == competing
    assert list(manifest.parent.iterdir()) == [manifest]
