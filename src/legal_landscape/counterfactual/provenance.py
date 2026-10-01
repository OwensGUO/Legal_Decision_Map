"""Bounded local checkpoint identity and immutable, atomic resume manifests.

Weight contents are deliberately not hashed: this is a reproducibility guard,
not a cryptographic verification of the complete checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

MANIFEST_NAME = "generator-provenance.json"
_METADATA_LIMIT = 64 * 1024 * 1024
_RECOVERY = (
    "Use a new OUTPUT_ROOT or deliberate artifact migration/removal; never relabel old data."
)


def _checkpoint_fingerprint(path: Path) -> str:
    try:
        weights = sorted({*path.glob("*.safetensors"), *path.glob("pytorch_model*.bin")})
        if not weights or any(not item.is_file() or item.stat().st_size == 0 for item in weights):
            raise ValueError("missing or empty Hugging Face weight files")
        metadata = sorted(
            {
                *path.glob("*.json"),
                *path.glob("*.jinja"),
                *path.glob("*.model"),
                *path.glob("*.tiktoken"),
                *path.glob("vocab.txt"),
                *path.glob("merges.txt"),
            }
        )
        if sum(item.stat().st_size for item in metadata) > _METADATA_LIMIT:
            raise ValueError("identity metadata exceeds the 64 MiB bound")
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("config.json must be an object")
        for index in path.glob("*.index.json"):
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"invalid weight_map in {index.name}")
            inventory = {item.name for item in weights}
            if not set(weight_map.values()) <= inventory:
                raise ValueError(f"missing weight shard referenced by {index.name}")
        material = {
            "method": "hf-metadata-and-weight-inventory-v1",
            "metadata": [
                [item.name, hashlib.sha256(item.read_bytes()).hexdigest()] for item in metadata
            ],
            "weights": [[item.name, item.stat().st_size] for item in weights],
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot establish checkpoint identity at {path}: {exc}") from exc


def generator_identity(config: Mapping[str, Any], *, mock: bool = False) -> dict[str, Any]:
    """Use the effective config, separating HTTP model alias from checkpoint path."""
    selector = str(config.get("selector", "")).strip()
    if not selector:
        raise ValueError("generator.selector is required for checkpoint identity")
    if mock:
        # Hash the implementation so a modified mock cannot resume older mock artifacts.
        fingerprint = hashlib.sha256(
            Path(__file__).with_name("generate.py").read_bytes()
        ).hexdigest()
        return {
            "schema_version": 1,
            "generator_selector": selector,
            "backend": "mock",
            "served_model_name": "mock-v1",
            "checkpoint_path": "builtin:mock-v1",
            "configured_revision": "mock-v1",
            "checkpoint_fingerprint": fingerprint,
            "fingerprint_method": "mock-implementation-sha256-v1",
        }
    for key in ("model_path", "checkpoint_path", "model_revision"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"generator.{key} is required for checkpoint identity")
    path = Path(config["checkpoint_path"]).expanduser().resolve()
    return {
        "schema_version": 1,
        "generator_selector": selector,
        "backend": config.get("backend", "vllm"),
        "served_model_name": config["model_path"],
        "checkpoint_path": str(path),
        "configured_revision": config["model_revision"],
        "checkpoint_fingerprint": _checkpoint_fingerprint(path),
        "fingerprint_method": "hf-metadata-and-weight-inventory-v1",
    }


def ensure_manifest(
    manifest: str | Path,
    identity: Mapping[str, Any],
    *,
    artifact_roots: Iterable[str | Path],
) -> None:
    """Validate existing provenance or atomically publish it for a fresh namespace.

    A hard link publishes the complete temporary file without replacing an existing
    manifest, including one created by a competing invocation. No model files change.
    """
    destination = Path(manifest)
    if destination.exists():
        try:
            persisted = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unreadable generator provenance {destination}. {_RECOVERY}") from exc
        if persisted != identity:
            raise ValueError(f"Generator provenance mismatch at {destination}. {_RECOVERY}")
        return
    for root in map(Path, artifact_roots):
        if root.exists() and (not root.is_dir() or any(root.rglob("*"))):
            raise ValueError(
                f"Missing generator provenance alongside artifacts at {root}. {_RECOVERY}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=".provenance-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(dict(identity), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            ensure_manifest(destination, identity, artifact_roots=())
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
