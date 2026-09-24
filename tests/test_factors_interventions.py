from __future__ import annotations

from legal_landscape.counterfactual.interventions import propose_interventions
from legal_landscape.factors.extract import extract_factors
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


def test_negated_mitigation_is_not_marked_true() -> None:
    factors = extract_factors("被告人拒不认罪，未退赃，不构成自首。", target_defendant="被告人")
    assert factors.confession is False
    assert factors.restitution is False
    assert factors.surrender is False


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
        charge="盗窃",
        factors=factors,
        charge_boundaries=(("盗窃", "诈骗"),),
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
        charge="故意伤害",
        factors=factors,
        charge_boundaries=(),
    )
    [ranked] = [item for item in specs if item.intervention_type == "sentence_rank"]
    assert ranked.target_factors.restitution is False
    assert ranked.rank_direction == 1
    assert "故意伤害" in ranked.rule_id


def test_charge_in_several_boundary_groups_flips_to_every_neighbour() -> None:
    specs = propose_interventions(
        parent_case_id="c1",
        defendant="某甲",
        charge="盗窃",
        factors=LegalFactors(conduct="秘密窃取"),
        charge_boundaries=(("盗窃", "诈骗"), ("盗窃", "抢夺", "抢劫")),
    )
    flips = [item for item in specs if item.intervention_type == "charge_flip"]
    assert [item.target_charge for item in flips] == ["诈骗", "抢夺", "抢劫"]
    assert len({item.rule_id for item in flips}) == 3
