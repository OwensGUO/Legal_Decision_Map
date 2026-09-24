from __future__ import annotations

import json

import pytest

from legal_landscape.counterfactual.generate import (
    GenerationRequest,
    MockGenerator,
    VLLMHTTPGenerator,
    generate_records,
)
from legal_landscape.counterfactual.validators import validate_generation
from legal_landscape.factors.schema import InterventionSpec, LegalFactors


def _spec(kind: str = "sentence_rank") -> InterventionSpec:
    source = LegalFactors(
        amount=1000.0,
        surrender=False,
        restitution=False,
        confession=True,
        role="principal",
        conduct="秘密窃取",
    )
    if kind == "charge_flip":
        target = LegalFactors(**{**source.to_dict(), "conduct": "虚构事实使被害人自愿交付"})
        return InterventionSpec(
            "c1", "某甲", "charge_flip", source, target, "诈骗", None, ("conduct",), "r1"
        )
    target = LegalFactors(**{**source.to_dict(), "restitution": True})
    return InterventionSpec(
        "c1", "某甲", "sentence_rank", source, target, None, -1, ("restitution",), "r2"
    )


def test_mock_generator_is_deterministic_valid_json() -> None:
    generator = MockGenerator()
    raw_a = generator.generate("某甲秘密取得财物。", _spec(), seed=42)
    raw_b = generator.generate("某甲秘密取得财物。", _spec(), seed=42)
    assert raw_a == raw_b
    payload = json.loads(raw_a)
    assert payload["target_defendant"] == "某甲"
    assert payload["realized_factors"]["restitution"] is True


def test_validator_accepts_target_change_and_preserved_non_targets() -> None:
    parent = "某甲秘密取得财物，涉案1000元并认罪。"
    raw = MockGenerator().generate(parent, _spec(), seed=42)
    result = validate_generation(parent, _spec(), raw)
    assert result.valid, result.errors
    assert result.counterfactual_text is not None


def test_validator_rejects_identity_label_sentence_leakage_and_unchanged_text() -> None:
    spec = _spec("charge_flip")
    leaked = json.dumps(
        {
            "counterfactual_text": "乙秘密取得财物，构成诈骗罪，应判处有期徒刑12个月。",
            "target_defendant": "乙",
            "changed_fields": ["conduct"],
            "realized_factors": spec.target_factors.to_dict(),
        },
        ensure_ascii=False,
    )
    result = validate_generation("某甲秘密取得财物。", spec, leaked)
    assert {"identity_not_preserved", "target_label_leakage", "sentence_leakage"} <= set(
        result.errors
    )

    duplicate = json.dumps(
        {
            "counterfactual_text": "某甲秘密取得财物。",
            "target_defendant": "某甲",
            "changed_fields": ["conduct"],
            "realized_factors": spec.target_factors.to_dict(),
        },
        ensure_ascii=False,
    )
    result = validate_generation("某甲秘密取得财物。", spec, duplicate)
    assert "unchanged" in result.errors


def test_generate_records_persists_provenance_and_resumes(tmp_path) -> None:
    output = tmp_path / "generated.jsonl"
    request = GenerationRequest("某甲秘密取得财物，涉案1000元并认罪。", _spec())
    records = generate_records(
        [request],
        MockGenerator(),
        output,
        model_revision="mock-v1",
        prompt_version="v1",
        sampling={"temperature": 0.0},
        seed=42,
        retries=1,
    ).records
    assert len(records) == 1
    record = records[0]
    assert record["parent_case_id"] == "c1"
    assert record["source_factors"]["amount"] == 1000.0
    assert record["model_revision"] == "mock-v1"
    assert record["retry_count"] == 0
    assert record["validation"]["valid"] is True
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1

    repeated = generate_records(
        [request],
        MockGenerator(),
        output,
        model_revision="mock-v1",
        prompt_version="v1",
        sampling={},
        seed=42,
        retries=1,
        resume=True,
    )
    assert repeated.records == []
    assert repeated.skipped_completed == 1
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_generate_records_retries_invalid_output(tmp_path) -> None:
    class BrokenGenerator:
        calls = 0

        def generate(self, parent_text, spec, *, seed):
            self.calls += 1
            return "not-json"

    generator = BrokenGenerator()
    records = generate_records(
        [GenerationRequest("某甲事实。", _spec())],
        generator,
        tmp_path / "failed.jsonl",
        model_revision="broken",
        prompt_version="v1",
        sampling={},
        seed=1,
        retries=2,
    ).records
    assert generator.calls == 3
    assert records[0]["validation"]["valid"] is False
    assert records[0]["retry_count"] == 2


def test_http_generator_rejects_remote_endpoint_without_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="allow_remote"):
        VLLMHTTPGenerator("https://example.com/v1/chat/completions", model="local")
    generator = VLLMHTTPGenerator(
        "https://example.com/v1/chat/completions", model="local", allow_remote=True
    )
    assert generator.endpoint.startswith("https://example.com")


def _amount_spec(amount: float) -> InterventionSpec:
    source = LegalFactors(amount=round(amount / 1.01, 2), conduct="秘密窃取")
    target = LegalFactors(amount=amount, conduct="秘密窃取")
    return InterventionSpec(
        "c1", "某甲", "invariant", source, target, None, None, ("amount",), "r3"
    )


def _payload(text: str, spec: InterventionSpec) -> str:
    return json.dumps(
        {
            "counterfactual_text": text,
            "target_defendant": spec.target_defendant,
            "changed_fields": list(spec.changed_fields),
            "realized_factors": spec.target_factors.to_dict(),
        },
        ensure_ascii=False,
    )


@pytest.mark.parametrize(
    ("amount", "written"),
    [(12468.45, "12468.45元"), (12468.45, "12,468.45元"), (1010000.0, "101万元")],
)
def test_amount_check_accepts_common_chinese_number_formats(amount: float, written: str) -> None:
    spec = _amount_spec(amount)
    parent = "某甲于2015年在某市秘密窃取他人财物，价值" + f"{spec.source_factors.amount}元。"
    text = "某甲于2015年在某市秘密窃取他人财物，价值" + written + "。"
    result = validate_generation(parent, spec, _payload(text, spec))
    assert result.valid, result.errors


def test_amount_check_rejects_unchanged_amount() -> None:
    spec = _amount_spec(10100.0)
    parent = "某甲于2015年在某市秘密窃取他人财物，价值10000元，后被抓获。"
    text = "某甲于2015年在某市秘密窃取他人财物，价值10000元，后被当场抓获。"
    result = validate_generation(parent, spec, _payload(text, spec))
    assert "target_change_missing" in result.errors


def test_long_faithful_rewrite_is_not_rejected_as_duplicate() -> None:
    spec = _amount_spec(10100.0)
    parent = "某甲于2015年3月在某市某区秘密窃取他人财物，价值10000元。" * 20
    text = parent.replace("价值10000元", "价值10100元", 1)
    result = validate_generation(parent, spec, _payload(text, spec))
    assert result.valid, result.errors


def test_excessive_rewrite_is_rejected() -> None:
    spec = _amount_spec(10100.0)
    parent = "某甲于2015年3月在某市某区秘密窃取他人财物，价值10000元。" * 5
    text = "某甲涉案10100元。"
    result = validate_generation(parent, spec, _payload(text, spec))
    assert "excessive_rewrite" in result.errors


def test_gambling_flip_conduct_is_not_label_leakage() -> None:
    source = LegalFactors(conduct="组织并经营赌博场所")
    target = LegalFactors(conduct="参与赌博")
    spec = InterventionSpec(
        "c1", "某甲", "charge_flip", source, target, "赌博", None, ("conduct",), "r4"
    )
    parent = "某甲在其家中组织并经营赌博场所，抽头渔利。"
    text = "某甲在其家中参与赌博，输赢数额较大。"
    result = validate_generation(parent, spec, _payload(text, spec), min_similarity=0.3)
    assert "target_label_leakage" not in result.errors
    assert result.valid, result.errors
    leaked = validate_generation(
        parent, spec, _payload(text + "其行为构成赌博罪。", spec), min_similarity=0.3
    )
    assert "target_label_leakage" in leaked.errors


def test_restitution_check_ignores_negated_mentions() -> None:
    source = LegalFactors(restitution=False)
    target = LegalFactors(restitution=True)
    spec = InterventionSpec(
        "c1", "某甲", "sentence_rank", source, target, None, -1, ("restitution",), "r5"
    )
    parent = "某甲秘密窃取他人财物，案发后未退赃。"
    result = validate_generation(
        parent,
        spec,
        _payload("某甲秘密窃取他人财物，案发后未退赃，后被抓获。", spec),
    )
    assert "target_change_missing" in result.errors
    result = validate_generation(
        parent, spec, _payload("某甲秘密窃取他人财物，案发后已全部退赃。", spec)
    )
    assert result.valid, result.errors


def test_concurrent_generation_writes_every_request_once(tmp_path) -> None:
    requests = [
        GenerationRequest(
            f"某甲秘密取得财物，涉案1000元并认罪。案件编号{index}。",
            InterventionSpec.from_dict({**_spec().to_dict(), "parent_case_id": f"c{index}"}),
        )
        for index in range(20)
    ]
    output = tmp_path / "concurrent.jsonl"
    result = generate_records(
        requests + requests[:3],
        MockGenerator(),
        output,
        model_revision="mock-v1",
        prompt_version="v2",
        sampling={},
        seed=42,
        retries=0,
        concurrency=8,
    )
    assert result.skipped_duplicate == 3
    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert sorted(item["generation_id"] for item in lines) == sorted(
        request.generation_id for request in requests
    )
    assert all(item["validation"]["valid"] for item in lines)


def test_transport_failures_are_not_written_and_rejections_are(tmp_path) -> None:
    from legal_landscape.counterfactual.generate import (
        GenerationRejectedError,
        GenerationTransportError,
    )

    class FlakyGenerator:
        def generate(self, parent_text, spec, *, seed):
            if spec.parent_case_id == "down":
                raise GenerationTransportError("connection refused")
            raise GenerationRejectedError("HTTP 400: prompt too long")

    down = GenerationRequest(
        "某甲事实。", InterventionSpec.from_dict({**_spec().to_dict(), "parent_case_id": "down"})
    )
    rejected = GenerationRequest("某甲事实。", _spec())
    output = tmp_path / "flaky.jsonl"
    result = generate_records(
        [down, rejected],
        FlakyGenerator(),
        output,
        model_revision="m",
        prompt_version="v2",
        sampling={},
        seed=1,
        retries=2,
        concurrency=2,
    )
    assert [item["generation_id"] for item in result.failed] == [down.generation_id]
    [record] = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert record["generation_id"] == rejected.generation_id
    assert record["validation"]["valid"] is False
    assert record["validation"]["errors"][0].startswith("request_rejected")


def test_http_generator_retries_server_errors_and_rejects_client_errors() -> None:
    import httpx

    from legal_landscape.counterfactual.generate import GenerationRejectedError

    statuses = iter([503, 200, 400])
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content))
        status = next(statuses)
        if status == 200:
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
        return httpx.Response(status, text="error")

    generator = VLLMHTTPGenerator(
        "http://127.0.0.1:30000/v1/chat/completions", model="local", backoff_seconds=0.0
    )
    generator._client = httpx.Client(transport=httpx.MockTransport(handler))
    assert generator.generate("某甲事实。", _spec(), seed=1) == "{}"
    assert len(seen_payloads) == 2
    assert seen_payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
    with pytest.raises(GenerationRejectedError):
        generator.generate("某甲事实。", _spec(), seed=1)
    generator.close()
