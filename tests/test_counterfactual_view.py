from __future__ import annotations

import json
from dataclasses import replace

import pytest

from legal_landscape.counterfactual.generate import (
    GenerationRequest,
    MockGenerator,
    generate_records,
)
from legal_landscape.counterfactual.provenance import generator_identity
from legal_landscape.factors.schema import InterventionSpec, LegalFactors


@pytest.fixture
def cache(tmp_path):
    identity = generator_identity({"selector": "mock"}, mock=True)
    spec = InterventionSpec(
        "c0", "某甲", "sentence_rank", LegalFactors(restitution=False),
        LegalFactors(restitution=True), None, -1, ("restitution",), "refund",
    )
    requests = [
        GenerationRequest("某甲取得他人财物，案发后未退赃。", replace(spec, parent_case_id=f"c{i}"))
        for i in range(13)
    ]
    path = tmp_path / "cache.jsonl"
    generate_records(
        requests, MockGenerator(), path, generator_identity=identity,
        prompt_version="v1", sampling={}, seed=42, retries=0,
    )
    return requests, path, tmp_path / "views/current.jsonl", identity


def test_view_selects_current_twelve_requests_without_modifying_historical_cache(cache):
    from legal_landscape.counterfactual.view import publish_request_view
    from legal_landscape.training.train import load_counterfactual_pairs

    requests, path, view, identity = cache
    before = path.read_bytes()
    current = list(reversed(requests[:12]))
    report = publish_request_view(
        current, path, view, generator_identity=identity, prompt_version="v1",
    )

    assert path.read_bytes() == before
    rows = [json.loads(line) for line in view.read_text().splitlines()]
    assert [row["generation_id"] for row in rows] == [r.generation_id for r in current]
    assert report == {"rows": 12, "valid": 12, "excluded": 1}
    parents = {r.spec.parent_case_id: {"fact_conservative": r.parent_text} for r in current}
    assert len(load_counterfactual_pairs(view, parents, filter_valid=True)) == 12
    assert view.stat().st_mtime_ns > 0


@pytest.mark.parametrize(
    "damage", ["duplicate", "missing", "identity", "spec", "json", "invalid_schema"]
)
def test_view_rejects_invalid_cache_and_preserves_previous_view(cache, damage):
    from legal_landscape.counterfactual.view import publish_request_view

    requests, path, view, identity = cache
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = next(row for row in rows if row["generation_id"] == requests[0].generation_id)
    if damage == "duplicate":
        rows.append(selected)
    elif damage == "missing":
        rows.remove(selected)
    elif damage == "identity":
        selected["generator_identity"] = {**identity, "checkpoint_fingerprint": "other"}
    elif damage == "spec":
        selected["target_factors"]["restitution"] = False
    elif damage == "invalid_schema":
        selected["validation"] = {"valid": False, "parsed": ["malformed"]}
    content = "".join(json.dumps(row) + "\n" for row in rows)
    if damage == "json":
        content += "broken JSON\n"
    path.write_text(content)
    view.parent.mkdir()
    view.write_text("previous view\n")
    before = path.read_bytes()

    with pytest.raises(ValueError):
        publish_request_view(
            requests[:1], path, view, generator_identity=identity, prompt_version="v1",
        )

    assert path.read_bytes() == before
    assert view.read_text() == "previous view\n"


def test_view_publication_failure_preserves_previous_view_and_removes_temp(cache, monkeypatch):
    from legal_landscape.counterfactual import view as view_module

    requests, path, view, identity = cache
    view.parent.mkdir()
    view.write_text("previous view\n")

    def fail_replace(*args):
        raise OSError("publication failed")

    monkeypatch.setattr(view_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failed"):
        view_module.publish_request_view(
            requests[:1], path, view, generator_identity=identity, prompt_version="v1",
        )

    assert view.read_text() == "previous view\n"
    assert list(view.parent.iterdir()) == [view]


@pytest.mark.parametrize("parsed_none", [False, True])
def test_view_preserves_invalid_generations_for_unfiltered_ablation(cache, parsed_none):
    from legal_landscape.counterfactual.view import publish_request_view
    from legal_landscape.training.train import load_counterfactual_pairs

    requests, path, view, identity = cache
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = next(row for row in rows if row["generation_id"] == requests[0].generation_id)
    selected["validation"]["valid"] = False
    if parsed_none:
        selected["validation"]["parsed"] = None
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = publish_request_view(
        requests[:1], path, view, generator_identity=identity, prompt_version="v1",
    )
    assert report["valid"] == 0
    assert json.loads(view.read_text())["validation"]["valid"] is False
    parents = {requests[0].spec.parent_case_id: {"fact_conservative": requests[0].parent_text}}
    assert len(load_counterfactual_pairs(view, parents, filter_valid=False)) == 1
    assert load_counterfactual_pairs(view, parents, filter_valid=True) == []
