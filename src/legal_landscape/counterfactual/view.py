"""Publish a validated current-request view without changing the generation cache."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from legal_landscape.counterfactual.generate import GenerationRequest


def publish_request_view(
    requests: list[GenerationRequest],
    cache_path: str | Path,
    view_path: str | Path,
    *,
    generator_identity: dict[str, Any],
    prompt_version: str,
) -> dict[str, int]:
    """Select exactly the requested IDs, validate them, then atomically publish JSONL.

    Historical rows remain in the cache. Selected duplicates or incompatible records are
    errors rather than an implicit choice between two versions of a training sample.
    """
    cache = Path(cache_path)
    destination = Path(view_path)
    if cache.resolve() == destination.resolve():
        raise ValueError("request view must use a different path from the generation cache")
    wanted: dict[str, GenerationRequest] = {}
    for request in requests:
        previous = wanted.setdefault(request.generation_id, request)
        if previous != request:
            raise ValueError(f"conflicting current requests for {request.generation_id}")
    selected: dict[str, str] = {}
    excluded = valid = 0
    with cache.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                generation_id = row["generation_id"]
                if not isinstance(generation_id, str) or not generation_id:
                    raise ValueError("missing generation ID")
                if generation_id not in wanted:
                    excluded += 1
                    continue
                if generation_id in selected:
                    raise ValueError(f"duplicate selected generation ID {generation_id}")
                if row.get("generator_identity") != generator_identity:
                    raise ValueError("generator identity mismatch")
                if row.get("model_revision") != generator_identity["checkpoint_fingerprint"]:
                    raise ValueError("checkpoint fingerprint mismatch")
                if row.get("prompt_version") != prompt_version:
                    raise ValueError("prompt version mismatch; use a new cache output path")
                spec = wanted[generation_id].spec.to_dict()
                for key, value in spec.items():
                    if row.get(key) != value:
                        raise ValueError(f"current intervention mismatch in {key}")
                validation = row["validation"]
                if not isinstance(validation, dict) or not isinstance(
                    validation.get("valid"), bool
                ):
                    raise ValueError("invalid validation schema")
                parsed = validation.get("parsed")
                if parsed is not None and not isinstance(parsed, dict):
                    raise ValueError("invalid parsed validation schema")
                if validation["valid"]:
                    if not isinstance(parsed, dict) or not isinstance(
                        parsed.get("counterfactual_text"), str
                    ) or not parsed["counterfactual_text"]:
                        raise ValueError("valid record has no counterfactual text")
                    valid += 1
                selected[generation_id] = line.rstrip("\n") + "\n"
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"invalid counterfactual cache at {cache}:{number}: {exc}"
                ) from exc
    missing = wanted.keys() - selected.keys()
    if missing:
        example = sorted(missing)[0]
        raise ValueError(
            f"counterfactual cache is missing {len(missing)} current requests: {example}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=".request-view-", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for generation_id in wanted:
                handle.write(selected[generation_id])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"rows": len(selected), "valid": valid, "excluded": excluded}
