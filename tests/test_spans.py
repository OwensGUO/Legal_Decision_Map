"""Sentence-span realization: deterministic splices, quoted flips, and their records."""

from __future__ import annotations

import json

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


def _flip_spec_named(defendant: str, co: tuple[str, ...]) -> InterventionSpec:
    return InterventionSpec(
        "c1", defendant, "charge_flip",
        LegalFactors(conduct="秘密窃取"),
        LegalFactors(conduct="虚构事实使被害人自愿交付"),
        "诈骗", None, ("conduct",), "boundary", co,
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


def test_flip_plan_quotes_the_whole_sentence() -> None:
    """The rewrite unit is the sentence, so the quote carries its terminator."""
    plan = plan_realization("张三窃取他人财物，涉案金额5000元。无人报警。", _flip_spec())
    assert plan.kind == "span"
    assert plan.quote == "张三窃取他人财物，涉案金额5000元。"


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


def test_multi_defendant_sentence_is_quoted_whole_so_neighbours_survive() -> None:
    parent = "张三与李四共同盗窃，张三窃取现金3000元，李四负责望风。"
    spec = _flip_spec_named("张三", ("李四",))
    plan = plan_realization(parent, spec)
    assert plan.kind == "span"
    assert plan.quote == parent
    replacement = "张三与李四共同盗窃，张三虚构事实骗取现金3000元，李四负责望风"
    assert replacement_problem(spec, parent, replacement) is None
    spliced, _ = splice(plan.base_text, plan.span, replacement)
    assert spliced == "张三与李四共同盗窃，张三虚构事实骗取现金3000元，李四负责望风。"


def test_multiple_source_clauses_are_refused_before_the_model_is_called() -> None:
    """A multi-count case describes the same conduct twice; one rewrite cannot fix both."""
    parent = "张三盗走被害人手机一部。张三又盗走被害人现金500元。"
    plan = plan_realization(parent, _flip_spec())
    assert plan.kind == "skip"
    assert plan.reason == "incomplete_flip"


def test_flip_without_any_source_clause_is_refused() -> None:
    spec = InterventionSpec(
        "c1", "张三", "charge_flip",
        LegalFactors(conduct="秘密窃取"),
        LegalFactors(conduct="虚构事实使被害人自愿交付"),
        "诈骗", None, ("conduct",), "boundary", ("李四",),
    )
    # The only clause names both actors, so no clause belongs to the target alone.
    plan = plan_realization("张三与李四共同实施行为。", spec)
    assert plan.kind == "skip"
    assert plan.reason == "no_target_span"


def test_flip_selects_the_sentence_that_describes_the_behaviour() -> None:
    parent = "被告人张三驾车窜至某小区，张三趁人不备夺走被害人手机。"
    plan = plan_realization(parent, _flip_spec())
    assert plan.kind == "span"
    assert plan.quote == parent


def test_flip_ignores_a_sentence_without_a_conduct_verb() -> None:
    parent = "被告人张三到案后如实供述。张三趁无人之机窃取被害人现金3000元。"
    plan = plan_realization(parent, _flip_spec())
    assert plan.kind == "span"
    assert plan.quote == "张三趁无人之机窃取被害人现金3000元。"


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


def test_bare_surname_leaves_the_actor_unattributable() -> None:
    """王破坏防护栏 cannot be pinned to 王某A: 王 also prefixes an unrelated 王某."""
    parent = (
        "被告人王某A与徐某A到小区，由徐敲门望风，"
        "王破坏防护栏进入被害人家中，将现金1400元窃取。"
    )
    plan = plan_realization(parent, _flip_spec_named("王某A", ("徐某A",)))
    assert plan.kind == "skip"
    assert plan.reason == "abbreviated_ambiguous"


def test_multi_count_taking_with_inflected_verbs_is_refused() -> None:
    """盗得 is the common judgment form and was invisible to the canonical cue table."""
    parent = "一、张三在刘某家中盗得戒指两枚，价值2304元；二、张三在张某家中盗得现金500元。"
    plan = plan_realization(parent, _flip_spec())
    assert plan.kind == "skip"
    assert plan.reason == "incomplete_flip"


def test_multi_count_refusal_ignores_a_summary_line() -> None:
    """共参与作案8次 states the count; it must not be treated as a rewriteable clause."""
    parent = "以上被告人张三共参与盗窃作案8次。"
    plan = plan_realization(parent, _flip_spec())
    assert plan.kind == "skip"
    assert plan.reason == "no_target_span"


def test_participation_only_clause_is_not_a_flip_target() -> None:
    parent = "被告人张三负责驾车接应，李四入室取财。以上事实清楚。"
    plan = plan_realization(parent, _flip_spec_named("张三", ("李四",)))
    assert plan.kind == "skip"
    assert plan.reason == "no_target_span"


def test_single_taking_sentence_is_still_selected() -> None:
    parent = "被告人张三趁人不备夺走被害人手机一部。"
    plan = plan_realization(parent, _flip_spec())
    assert plan.kind == "span"
    assert plan.quote == parent


def _harm_spec() -> InterventionSpec:
    return InterventionSpec(
        "c1", "张三", "charge_flip",
        LegalFactors(conduct="因个人纠纷定向伤害"),
        LegalFactors(conduct="无事生非随意殴打"),
        "寻衅滋事", None, ("conduct",), "boundary",
    )


def test_injury_grade_is_not_treated_as_the_source_conduct() -> None:
    """轻伤二级 is a medical fact about the victim and must survive a conduct flip."""
    parent = "被告人张三因琐事与被害人发生纠纷，后殴打被害人致轻伤二级。"
    plan = plan_realization(parent, _harm_spec())
    assert plan.kind == "span"
    assert plan.quote == parent
    replacement = "被告人张三无事生非，随意殴打被害人致轻伤二级"
    assert replacement_problem(_harm_spec(), parent, replacement) is None


def test_harm_flip_that_keeps_the_source_grounds_is_still_refused() -> None:
    parent = "被告人张三因琐事与被害人发生纠纷，后殴打被害人致轻伤二级。"
    problem = replacement_problem(
        _harm_spec(), parent, "被告人张三因琐事随意殴打被害人致轻伤二级"
    )
    assert problem == "replacement_keeps_source_conduct"
