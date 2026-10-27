from __future__ import annotations

import pytest

from legal_landscape.counterfactual.interventions import propose_interventions
from legal_landscape.factors.extract import conduct_cues, extract_factors
from legal_landscape.factors.schema import LegalFactors
from legal_landscape.factors.validate import validate_factors


def test_extracts_supported_legal_factors_from_literal_fact() -> None:
    factors = extract_factors(
        "被告人系主犯，涉案金额2万元，案发后主动投案自首，如实供述并认罪，已全部退赃。",
        target_defendant="被告人",
    )
    assert factors.amount == 20000.0
    assert factors.surrender is True
    assert factors.restitution is True
    assert factors.confession is True
    assert factors.role == "principal"
    assert validate_factors(factors) == ()


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ("未如实供述", "confession"),
        ("未主动投案", "surrender"),
        ("未退赔", "restitution"),
        ("没有退赃", "restitution"),
        ("未能退赔", "restitution"),
        ("尚未退赃", "restitution"),
        ("拒不退赔", "restitution"),
    ],
)
def test_negated_factor_cues_are_false_rather_than_affirmative(text, field) -> None:
    factors = extract_factors(f"被告人甲{text}。", target_defendant="甲")
    assert getattr(factors, field) is False


def test_untui_and_untuipei_have_the_same_state() -> None:
    stolen = extract_factors("被告人甲未退赃。", target_defendant="甲")
    compensated = extract_factors("被告人甲未退赔。", target_defendant="甲")
    assert stolen.restitution is False
    assert compensated.restitution is False


def test_conflicting_restitution_cues_are_unknown() -> None:
    factors = extract_factors("被告人甲未退赃，但已退赔。", target_defendant="甲")
    assert factors.restitution is None


def test_negated_mitigation_is_not_marked_true() -> None:
    factors = extract_factors("被告人拒不认罪，未退赃，不构成自首。", target_defendant="被告人")
    assert factors.confession is False
    assert factors.restitution is False
    assert factors.surrender is False


def test_charge_names_alone_never_produce_a_conduct_factor() -> None:
    factors = extract_factors(
        "被告人甲寻衅滋事、故意伤害、诈骗、抢夺、开设赌场、暴力抢劫。", target_defendant="甲"
    )
    assert factors.conduct is None


@pytest.mark.parametrize(
    ("text", "conduct"),
    [
        ("甲谎称能办理贷款，骗取被害人钱款。", "虚构事实使被害人自愿交付"),
        ("甲趁人不备夺走被害人手机。", "趁人不备公开夺取"),
        ("甲持刀威胁被害人，劫取现金。", "暴力胁迫取财"),
        ("甲提供场所组织赌博并抽头渔利。", "组织并经营赌博场所"),
        ("甲因琐事与被害人发生纠纷，致其轻伤二级。", "因个人纠纷定向伤害"),
        ("甲借故生非，随意殴打路人。", "无事生非随意殴打"),
    ],
)
def test_conduct_is_extracted_from_behaviour_descriptions(text: str, conduct: str) -> None:
    assert extract_factors(text, target_defendant="甲").conduct == conduct


def test_conduct_cues_report_every_described_conduct() -> None:
    assert conduct_cues("甲盗走手机后谎称手机丢失，骗取保险金。") == {
        "秘密窃取",
        "虚构事实使被害人自愿交付",
    }


def test_factors_ignore_sentences_about_other_defendants_only() -> None:
    text = "被告人甲主动投案自首。被告人乙被抓获后拒不认罪。二人共同盗走财物价值5000元。"
    second = extract_factors(text, target_defendant="乙", co_defendants=("甲",))
    assert second.surrender is None
    assert second.confession is False
    assert second.amount == 5000.0
    first = extract_factors(text, target_defendant="甲", co_defendants=("乙",))
    assert first.surrender is True
    assert first.confession is None


def _invariant(amount: float):
    [spec] = [
        item
        for item in propose_interventions(
            parent_case_id="c1",
            defendant="某甲",
            charges=("盗窃",),
            factors=LegalFactors(amount=amount),
            charge_boundaries=(),
        )
        if item.intervention_type == "invariant"
    ] or [None]
    return spec


def test_invariant_amount_change_stays_inside_the_statutory_amount_band() -> None:
    assert _invariant(10000.0).target_factors.amount == 10100.0
    assert _invariant(2990.0).target_factors.amount == 2960.1
    assert _invariant(3000.0).target_factors.amount == 3030.0


def test_charge_flip_requires_a_single_charge_parent() -> None:
    specs = propose_interventions(
        parent_case_id="c1",
        defendant="某甲",
        charges=("盗窃", "故意伤害"),
        factors=LegalFactors(amount=1000.0, conduct="秘密窃取"),
        charge_boundaries=(("盗窃", "诈骗"),),
    )
    assert "charge_flip" not in {item.intervention_type for item in specs}
    assert all(item.rule_id.startswith("盗窃+故意伤害:") for item in specs)


def test_intervention_builder_produces_all_typed_rules() -> None:
    factors = LegalFactors(
        amount=10000.0,
        surrender=False,
        restitution=False,
        confession=True,
        role="principal",
        conduct="秘密窃取",
    )
    specs = propose_interventions(
        parent_case_id="c1",
        defendant="某甲",
        charges=("盗窃",),
        factors=factors,
        charge_boundaries=(("盗窃", "诈骗"),),
        penalty_type="fixed_term",
        imprisonment_months=12,
    )
    by_type = {item.intervention_type: item for item in specs}
    assert set(by_type) == {"invariant", "charge_flip", "sentence_rank"}
    assert by_type["charge_flip"].target_charge == "诈骗"
    assert by_type["charge_flip"].target_factors.conduct == "虚构事实使被害人自愿交付"
    assert by_type["sentence_rank"].rank_direction == -1
    assert by_type["sentence_rank"].changed_fields == ("restitution",)
    assert by_type["invariant"].rank_direction is None


def test_sentence_rule_is_conditional_and_reverses_when_mitigation_removed() -> None:
    factors = LegalFactors(restitution=True, confession=True, role="accessory")
    specs = propose_interventions(
        parent_case_id="c1",
        defendant="某甲",
        charges=("故意伤害",),
        factors=factors,
        charge_boundaries=(),
        penalty_type="fixed_term",
        imprisonment_months=24,
    )
    [ranked] = [item for item in specs if item.intervention_type == "sentence_rank"]
    assert ranked.target_factors.restitution is False
    assert ranked.rank_direction == 1
    assert "故意伤害" in ranked.rule_id


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"charges": ("盗窃", "诈骗")}, "not_single_charge"),
        ({"penalty_type": "life", "imprisonment_months": None}, "penalty_life"),
        ({"penalty_type": "death", "imprisonment_months": None}, "penalty_death"),
        ({"penalty_type": "exempt", "imprisonment_months": None}, "penalty_exempt"),
        ({"imprisonment_months": None}, "missing_months"),
        ({"factors": LegalFactors(restitution=None)}, "unknown_restitution"),
    ],
)
def test_sentence_rank_stays_out_of_ineligible_cases(override, reason) -> None:
    from legal_landscape.counterfactual.interventions import sentence_rank_exclusion

    arguments = {
        "parent_case_id": "c1",
        "defendant": "某甲",
        "charges": ("盗窃",),
        "factors": LegalFactors(restitution=True),
        "charge_boundaries": (),
        "penalty_type": "fixed_term",
        "imprisonment_months": 12,
    }
    arguments.update(override)
    specs = propose_interventions(**arguments)
    assert all(item.intervention_type != "sentence_rank" for item in specs)
    assert sentence_rank_exclusion(
        charges=arguments["charges"],
        penalty_type=arguments["penalty_type"],
        imprisonment_months=arguments["imprisonment_months"],
        restitution=arguments["factors"].restitution,
    ) == reason


def test_charge_in_several_boundary_groups_flips_to_every_neighbour() -> None:
    specs = propose_interventions(
        parent_case_id="c1",
        defendant="某甲",
        charges=("盗窃",),
        factors=LegalFactors(conduct="秘密窃取"),
        charge_boundaries=(("盗窃", "诈骗"), ("盗窃", "抢夺", "抢劫")),
    )
    flips = [item for item in specs if item.intervention_type == "charge_flip"]
    assert [item.target_charge for item in flips] == ["诈骗", "抢夺", "抢劫"]
    assert len({item.rule_id for item in flips}) == 3
