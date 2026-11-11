"""Regressions for validator-cache identity and case-level coverage."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from legal_landscape.counterfactual.generate import GenerationRequest, text_sha256
from legal_landscape.counterfactual.prompts import PROMPT_VERSION
from legal_landscape.counterfactual.validators import VALIDATOR_VERSION
from legal_landscape.data.sanitize import SANITIZER_VERSION
from legal_landscape.factors.schema import InterventionSpec, LegalFactors

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def _spec() -> InterventionSpec:
    return InterventionSpec(
        "c1",
        "某甲",
        "sentence_rank",
        LegalFactors(restitution=False),
        LegalFactors(restitution=True),
        None,
        -1,
        ("restitution",),
        "refund",
    )


def _legacy_generation_id(request: GenerationRequest) -> str:
    """Identity used before VALIDATOR_VERSION joined the cache key."""
    material = "\0".join(
        (
            request.spec.parent_case_id,
            request.spec.target_defendant,
            "\n".join(request.spec.co_defendants),
            request.spec.rule_id,
            text_sha256(request.parent_text),
            PROMPT_VERSION,
            SANITIZER_VERSION,
        )
    )
    return hashlib.sha256(material.encode()).hexdigest()[:24]


def test_validator_version_changes_the_current_request_identity() -> None:
    parent = "某甲取得他人财物，案发后未退赃。"
    request = GenerationRequest(parent, _spec())
    assert request.generation_id != _legacy_generation_id(request)
    material = "\0".join(
        (
            request.spec.parent_case_id,
            request.spec.target_defendant,
            "\n".join(request.spec.co_defendants),
            request.spec.rule_id,
            text_sha256(parent),
            PROMPT_VERSION,
            SANITIZER_VERSION,
            VALIDATOR_VERSION,
        )
    )
    assert request.generation_id == hashlib.sha256(material.encode()).hexdigest()[:24]


def test_resume_appends_current_validator_records_and_keeps_the_old_cache(tmp_path) -> None:
    parent = "某甲取得他人财物，案发后未退赃。"
    request = GenerationRequest(parent, _spec())
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps({"parent_text": parent, "spec": _spec().to_dict()}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cf" / "generated.jsonl"
    view = tmp_path / "cf" / "current.jsonl"
    cache.parent.mkdir()
    from legal_landscape.config import load_config
    from legal_landscape.counterfactual.provenance import (
        MANIFEST_NAME,
        ensure_manifest,
        generator_identity,
    )

    identity = generator_identity(
        load_config(ROOT / "configs/cf/qwen38_27b.yaml")["generator"], mock=True
    )
    ensure_manifest(cache.parent / MANIFEST_NAME, identity, artifact_roots=[cache.parent])
    legacy = {
        "generation_id": _legacy_generation_id(request),
        "parent_case_id": "c1",
        "target_defendant": "某甲",
        "prompt_version": PROMPT_VERSION,
        "sanitizer_version": SANITIZER_VERSION,
        "validator_version": "previous-protocol",
        "validation": {"valid": True, "errors": []},
        "counterfactual_strict": "旧缓存正文。",
    }
    cache.write_text(json.dumps(legacy, ensure_ascii=False) + "\n", encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_counterfactuals.py"),
        "--mock",
        "--execute",
        "--resume",
        "--input",
        str(source),
        "--output",
        str(cache),
        "--view-output",
        str(view),
        "--limit",
        "4",
    ]
    first = subprocess.run(command, env=_env(), capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    report = json.loads(first.stdout)
    assert report["generated"] == 1
    assert report["skipped_completed"] == 0
    lines = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    assert [row["generation_id"] for row in lines].count(legacy["generation_id"]) == 1
    assert [row["generation_id"] for row in lines].count(request.generation_id) == 1
    current = next(row for row in lines if row["generation_id"] == request.generation_id)
    assert current["validator_version"] == VALIDATOR_VERSION
    view_rows = [json.loads(line) for line in view.read_text(encoding="utf-8").splitlines()]
    assert [row["generation_id"] for row in view_rows] == [request.generation_id]
    assert all(row["validator_version"] == VALIDATOR_VERSION for row in view_rows)

    from legal_landscape.training.train import load_counterfactual_pairs

    parents = {
        "c1": {
            "target_defendant": "某甲",
            "fact_strict": parent,
            "charges": ["盗窃"],
            "penalty_type": "fixed_term",
            "imprisonment_months": 12,
            "factors": {"restitution": False},
        }
    }
    loaded = load_counterfactual_pairs(view, parents, filter_valid=True)
    assert len(loaded) == 1
    assert loaded[0]["counterfactual_factors"]["restitution"] is True

    before = cache.read_bytes()
    second = subprocess.run(command, env=_env(), capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr
    again = json.loads(second.stdout)
    assert again["generated"] == 0
    assert again["skipped_completed"] == 1
    assert cache.read_bytes() == before


def test_view_rejects_a_current_id_with_a_stale_validator_version(tmp_path) -> None:
    from legal_landscape.counterfactual.provenance import generator_identity
    from legal_landscape.counterfactual.view import publish_request_view

    parent = "某甲取得他人财物，案发后未退赃。"
    request = GenerationRequest(parent, _spec())
    identity = generator_identity({"selector": "mock"}, mock=True)
    cache = tmp_path / "cache.jsonl"
    view = tmp_path / "view.jsonl"
    cache.write_text(
        json.dumps(
            {
                "generation_id": request.generation_id,
                "parent_case_id": "c1",
                "target_defendant": "某甲",
                "intervention_type": "sentence_rank",
                "source_factors": _spec().source_factors.to_dict(),
                "target_factors": _spec().target_factors.to_dict(),
                "changed_fields": ["restitution"],
                "target_charge": None,
                "rank_direction": -1,
                "rule_id": "refund",
                "co_defendants": [],
                "model_revision": identity["checkpoint_fingerprint"],
                "generator_identity": identity,
                "prompt_version": PROMPT_VERSION,
                "sanitizer_version": SANITIZER_VERSION,
                "validator_version": "previous-protocol",
                "parent_text_sha256": text_sha256(parent),
                "validation": {"valid": True, "errors": [], "parsed": {}},
                "counterfactual_strict": "某甲已退赃。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        publish_request_view(
            [request],
            cache,
            view,
            generator_identity=identity,
            prompt_version=PROMPT_VERSION,
        )
    except ValueError as exc:
        assert "validator" in str(exc).lower()
    else:
        raise AssertionError("stale validator_version was published")
    assert cache.read_bytes()
    assert not view.exists()


def _parent(
    case_id: str,
    group_id: str,
    defendant: str,
    charges: list[str],
    fact: str,
    *,
    co_defendants: list[str] | None = None,
    amount: float | None = None,
    restitution: bool | None = False,
    penalty: str = "fixed_term",
    months: int | None = 12,
    conduct: str | None = None,
) -> dict:
    factors: dict = {"restitution": restitution}
    if amount is not None:
        factors["amount"] = amount
    if conduct is not None:
        factors["conduct"] = conduct
    return {
        "case_id": case_id,
        "group_id": group_id,
        "target_defendant": defendant,
        "co_defendants": co_defendants or [],
        "fact_strict": fact,
        "charges": charges,
        "factors": factors,
        "penalty_type": penalty,
        "imprisonment_months": months,
    }


def _run_coverage(tmp_path: Path, rows: list[dict], name: str) -> dict:
    source = tmp_path / name / "processed" / "train.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (source.parent / "metadata.json").write_text(
        json.dumps(
            {"sanitizer": {"version": SANITIZER_VERSION, "charge_lexicon": ["盗窃", "诈骗"]}}
        ),
        encoding="utf-8",
    )
    output = tmp_path / name / "cf" / "cache.jsonl"
    view = tmp_path / name / "cf" / "view.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_counterfactuals.py"),
            "--mock",
            "--execute",
            "--resume",
            "--input",
            str(source),
            "--output",
            str(output),
            "--view-output",
            str(view),
            "--limit",
            "20",
        ],
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_coverage_counts_one_case_when_two_defendants_share_a_group(tmp_path) -> None:
    shared = "甲乙共同窃取财物，价值1500元。"
    rows = [
        _parent(
            "g-a-d000", "g-a", "甲", ["盗窃"], shared,
            co_defendants=["乙"], amount=1500, conduct="秘密窃取",
        ),
        _parent(
            "g-a-d001", "g-a", "乙", ["盗窃"], shared + "乙参与。",
            co_defendants=["甲"], amount=1500, conduct="秘密窃取",
        ),
        _parent(
            "g-b-d000", "g-b", "丙", ["诈骗"], "丙谎称借款。",
            amount=4000, conduct="虚构事实使被害人自愿交付",
        ),
    ]
    report = _run_coverage(tmp_path, rows, "cases")
    parents = report["coverage"]["parents"]["before_filters"]
    assert parents["unit"] == "defendant_record"
    assert parents["count"] == 3
    assert parents["case_count"] == 2
    assert parents["defendants_per_case"]["unit"] == "case"
    assert parents["defendants_per_case"]["counts"] == {"2": 1, "1": 1}
    view = report["coverage"]["view"]
    assert view["unit"] == "generation_request"
    assert view["records"] > parents["count"]
    assert view["case_count"] == 2
    assert view["defendants_per_case"]["counts"] == {"2": 1, "1": 1}

    resumed = _run_coverage(tmp_path, rows, "cases")
    assert resumed["coverage"]["generation"]["new"] == 0
    assert resumed["coverage"]["view"]["records"] == view["records"]
    assert resumed["coverage"]["view"]["case_count"] == view["case_count"]
    assert resumed["coverage"]["view"]["defendants_per_case"] == view["defendants_per_case"]
    assert resumed["coverage"]["parents"] == report["coverage"]["parents"]


def test_partial_case_exclusion_keeps_case_size_and_record_denominators(tmp_path) -> None:
    rows = [
        _parent(
            "g-a-d000", "g-a", "甲", ["盗窃"], "甲秘密窃取财物。",
            co_defendants=["乙"], amount=1500, conduct="秘密窃取",
        ),
        _parent(
            "g-a-d001", "g-a", "乙", ["盗窃"], "长" * 5000,
            co_defendants=["甲"], amount=1500, conduct="秘密窃取",
        ),
    ]
    report = _run_coverage(tmp_path, rows, "partial")
    before = report["coverage"]["parents"]["before_filters"]
    after = report["coverage"]["parents"]["after_filters"]
    excluded = report["coverage"]["parents"]["excluded"]["length"]
    assert before["count"] == 2
    assert before["case_count"] == 1
    assert before["defendants_per_case"]["counts"] == {"2": 1}
    assert after["count"] == 1
    assert after["case_count"] == 1
    assert after["defendants_per_case"]["counts"] == {"2": 1}
    assert excluded["unit"] == "defendant_record"
    assert excluded["count"] == 1
    assert excluded["affected_cases"] == 1
    assert excluded["partial_cases"] == 1
