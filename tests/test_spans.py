"""Sentence-span realization: deterministic splices, quoted flips, and their records."""

from __future__ import annotations

import json

import pytest

from legal_landscape.counterfactual.generate import (
    GenerationRequest,
    MockGenerator,
    generate_records,
)
from legal_landscape.counterfactual.prompts import PROMPT_VERSION, build_messages
from legal_landscape.counterfactual.provenance import generator_identity
from legal_landscape.counterfactual.spans import (
    mask_counts,
    mentions,
    plan_realization,
    replacement_problem,
    scope_spans,
    sentence_spans,
    splice,
)
from legal_landscape.counterfactual.validators import validate_generation
from legal_landscape.factors.schema import InterventionSpec, LegalFactors


def _restitution_spec(*, target: str = "张三", co: tuple[str, ...] = ()) -> InterventionSpec:
    return InterventionSpec(
        "c1", target, "sentence_rank",
        LegalFactors(restitution=False), LegalFactors(restitution=True),
        None, -1, ("restitution",), "refund", co,
    )


def _amount_spec(source: float, target: float, *, defendant: str = "张三") -> InterventionSpec:
    return InterventionSpec(
        "c1", defendant, "invariant",
        LegalFactors(amount=source, conduct="秘密窃取"),
        LegalFactors(amount=target, conduct="秘密窃取"),
        None, None, ("amount",), "amount",
    )


def _flip_spec(*, defendant: str = "张三") -> InterventionSpec:
    return InterventionSpec(
        "c1", defendant, "charge_flip",
        LegalFactors(conduct="秘密窃取"),
        LegalFactors(conduct="虚构事实使被害人自愿交付"),
        "诈骗", None, ("conduct",), "boundary",
    )


def test_sentence_spans_tile_the_text_exactly() -> None:
    text = "张三窃取手机一部。李四在场；无人报警。"
    spans = sentence_spans(text)
    assert "".join(span.text for span in spans) == text
    assert [text[span.start : span.end] for span in spans] == [span.text for span in spans]


def test_mentions_does_not_read_one_name_inside_a_longer_co_defendant() -> None:
    assert mentions("李某A到案。", "李某", ("李某A",)) is False
    assert mentions("李某到案。", "李某", ("李某A",)) is True


def test_scope_spans_drops_clauses_about_other_defendants_only() -> None:
    text = "张三与李四共同盗窃，张三分得300元，李四分得200元。"
    kept = [span.text for span in scope_spans(text, _restitution_spec(co=("李四",)))]
    # The shared opening clause names both actors, so only the target's own clause survives.
    assert kept == ["张三分得300元"]


def test_restitution_is_spliced_deterministically_and_validates() -> None:
    parent = "张三取得他人财物，案发后未退赃。"
    plan = plan_realization(parent, _restitution_spec())
    assert plan.kind == "deterministic"
    assert plan.text == "张三取得他人财物，案发后已退赃退赔。"
    result = validate_generation(
        parent,
        _restitution_spec(),
        json.dumps({"counterfactual_text": plan.text}, ensure_ascii=False),
    )
    assert result.valid, result.errors


def test_restitution_splice_keeps_attribution_with_co_defendants() -> None:
    parent = "张三未退赃。李四未退赃。"
    plan = plan_realization(parent, _restitution_spec(co=("李四",)))
    assert plan.kind == "deterministic"
    assert plan.text == "张三案发后已退赃退赔。李四未退赃。"
    result = validate_generation(
        parent,
        _restitution_spec(co=("李四",)),
        json.dumps({"counterfactual_text": plan.text}, ensure_ascii=False),
    )
    assert result.valid, result.errors


def test_amount_splice_writes_yuan_and_respects_the_cued_literal() -> None:
    plan = plan_realization("张三窃取他人财物，涉案金额5000元。", _amount_spec(5000.0, 5050.0))
    assert plan.kind == "deterministic"
    assert plan.text == "张三窃取他人财物，涉案金额5050元。"


def test_amount_splice_converts_wan_yuan_to_yuan() -> None:
    plan = plan_realization("张三窃取他人财物，涉案金额1.2万元。", _amount_spec(12000.0, 12120.0))
    assert plan.kind == "deterministic"
    assert plan.text == "张三窃取他人财物，涉案金额12120元。"


def test_structural_skip_needs_no_model_call() -> None:
    plan = plan_realization("案发经过不详。", _restitution_spec())
    assert plan.kind == "deterministic"
    plan = plan_realization("案发经过不详。", _amount_spec(1000.0, 1010.0))
    assert plan.kind == "skip"
    assert plan.reason == "no_amount_span"


def test_flip_plan_quotes_the_source_behaviour_sentence() -> None:
    plan = plan_realization("张三窃取他人财物，涉案金额5000元。无人报警。", _flip_spec())
    assert plan.kind == "span"
    assert plan.quote == "张三窃取他人财物"


@pytest.mark.parametrize(
    ("replacement", "problem"),
    [
        ("李四虚构事实骗取财物。", "replacement_missing_target"),
        ("张三李四虚构事实骗取财物。", "replacement_mentions_other_defendant"),
        ("张三窃取他人财物。", "replacement_keeps_source_conduct"),
        ("张三取走他人财物。", "replacement_missing_target_conduct"),
        ("", "empty_replacement"),
    ],
)
def test_replacement_problems_are_caught_before_splicing(replacement, problem) -> None:
    spec = InterventionSpec(
        "c1", "张三", "charge_flip",
        LegalFactors(conduct="秘密窃取"),
        LegalFactors(conduct="虚构事实使被害人自愿交付"),
        "诈骗", None, ("conduct",), "boundary", ("李四",),
    )
    assert replacement_problem(spec, "张三窃取他人财物。", replacement) == problem


def test_replacement_cannot_restore_a_masked_span() -> None:
    spec = _flip_spec()
    problem = replacement_problem(
        spec, "张三窃取手机，价值××元。", "张三虚构事实骗取手机，价值4450元。"
    )
    assert problem == "replacement_restores_mask"


def test_splice_preserves_the_terminator_and_reports_new_offsets() -> None:
    base = "张三窃取手机。李四在场。"
    span = sentence_spans(base)[0]
    spliced, new = splice(base, span, "张三虚构事实骗取手机")
    assert spliced == "张三虚构事实骗取手机。李四在场。"
    assert spliced[new.start : new.end] == "张三虚构事实骗取手机。"


def test_span_prompt_asks_for_one_sentence_and_hides_labels() -> None:
    system, user = build_messages("张三窃取手机。", _flip_spec(), quote="张三窃取手机。")
    payload = json.loads(user["content"])
    assert payload["source_sentence"] == "张三窃取手机。"
    assert payload["response_schema"] == {"replacement_sentence": "string"}
    for hidden in ("诈骗", "盗窃", "charge_flip", "rule_id", "target_charge"):
        assert hidden not in user["content"]
    assert "replacement_sentence" in system["content"]


def test_mask_counts_are_compared_per_mask() -> None:
    assert mask_counts("价值××元，[已屏蔽]") == (1, 1)


class _NoCallGenerator:
    """Deterministic requests must never reach the model."""

    def generate(self, parent_text, spec, *, seed, quote=None):
        raise AssertionError("deterministic realization called the model")


def test_generate_records_splices_without_any_model_call(tmp_path) -> None:
    request = GenerationRequest("张三取得他人财物，案发后未退赃。", _restitution_spec())
    result = generate_records(
        [request],
        _NoCallGenerator(),
        tmp_path / "det.jsonl",
        generator_identity=generator_identity({"selector": "mock"}, mock=True),
        prompt_version=PROMPT_VERSION,
        sampling={},
        seed=1,
        retries=2,
    )
    [record] = result.records
    assert record["validation"]["valid"] is True
    assert record["retry_count"] == 0
    assert record["counterfactual_strict"] == "张三取得他人财物，案发后已退赃退赔。"


def test_all_three_intervention_types_realize_end_to_end(tmp_path) -> None:
    spec = _flip_spec()
    requests = [
        GenerationRequest("张三取得他人财物，案发后未退赃。", _restitution_spec()),
        GenerationRequest("张三窃取他人财物，涉案金额5000元。", _amount_spec(5000.0, 5050.0)),
        GenerationRequest("张三窃取他人财物，涉案金额5000元。", spec),
    ]
    result = generate_records(
        requests,
        MockGenerator(),
        tmp_path / "all.jsonl",
        generator_identity=generator_identity({"selector": "mock"}, mock=True),
        prompt_version=PROMPT_VERSION,
        sampling={},
        seed=1,
        retries=0,
    )
    assert [record["validation"]["valid"] for record in result.records] == [True, True, True]
    assert [record["retry_count"] for record in result.records] == [0, 0, 0]


def test_multi_defendant_clause_is_edited_without_disturbing_the_neighbour() -> None:
    parent = "张三与李四共同盗窃，张三窃取现金3000元，李四负责望风。"
    spec = _flip_spec()
    plan = plan_realization(parent, spec)
    assert plan.kind == "span"
    assert plan.quote == "张三窃取现金3000元"
    spliced, _ = splice(
        plan.base_text, plan.span, "张三虚构事实骗取现金3000元"
    )
    assert spliced == "张三与李四共同盗窃，张三虚构事实骗取现金3000元，李四负责望风。"


def test_multiple_source_clauses_are_refused_before_the_model_is_called() -> None:
    """A multi-count case describes the same conduct twice; one rewrite cannot fix both."""
    parent = "张三盗走被害人手机一部。张三又盗走被害人现金500元。"
    plan = plan_realization(parent, _flip_spec())
    assert plan.kind == "skip"
    assert plan.reason == "incomplete_flip"


def test_flip_without_any_source_clause_is_refused() -> None:
    plan = plan_realization("张三与李四共同实施行为。", _flip_spec())
    assert plan.kind == "skip"
    assert plan.reason == "no_source_conduct_span"


def test_amount_splice_ignores_a_literal_that_is_not_the_source_amount() -> None:
    """A joint total or a compensation figure is a different fact from the case amount."""
    parent = "张三盗走财物，被盗财物共计价值7062元。案发后张三家属赔偿损失共计7132.62元。"
    plan = plan_realization(parent, _amount_spec(7062.0, 7132.62))
    assert plan.kind == "deterministic"
    assert plan.text is not None
    # Only the case amount moves; the pre-existing compensation figure stays untouched.
    assert "7062元" not in plan.text
    assert "被盗财物共计价值7132.62元" in plan.text
    assert "赔偿损失共计7132.62元" in plan.text


def test_amount_skip_when_only_a_different_figure_is_present() -> None:
    parent = "张三赔偿被害人共计7132.62元。"
    plan = plan_realization(parent, _amount_spec(7062.0, 7132.62))
    assert plan.kind == "skip"
    assert plan.reason == "no_amount_span"
