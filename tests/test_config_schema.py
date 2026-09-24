from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from legal_landscape.config import load_config
from legal_landscape.data.schema import CaseUnit, PenaltyType
from legal_landscape.factors.schema import InterventionSpec, LegalFactors


def test_case_unit_preserves_multilabel_and_excludes_it_from_main_regression() -> None:
    case = CaseUnit(
        dataset="cail",
        split="train",
        case_id="c1",
        group_id="c1",
        target_defendant="段某",
        fact_raw="事实",
        charges=("故意伤害", "寻衅滋事"),
        penalty_type=PenaltyType.FIXED_TERM.value,
        imprisonment_months=12,
        source_path="data.json:1",
    )
    assert case.charges == ("故意伤害", "寻衅滋事")
    assert not case.is_sentence_regression_eligible
    with pytest.raises(FrozenInstanceError):
        case.case_id = "changed"  # type: ignore[misc]


def test_single_charge_fixed_term_is_sentence_regression_eligible() -> None:
    case = CaseUnit(
        dataset="cail",
        split="train",
        case_id="c1",
        group_id="c1",
        target_defendant="段某",
        fact_raw="事实",
        charges=("故意伤害",),
        penalty_type=PenaltyType.FIXED_TERM.value,
        imprisonment_months=12,
        source_path="data.json:1",
    )
    assert case.is_sentence_regression_eligible


@pytest.mark.parametrize("penalty", [PenaltyType.LIFE.value, PenaltyType.DEATH.value])
def test_life_and_death_reject_fabricated_months(penalty: str) -> None:
    with pytest.raises(ValueError, match="months"):
        CaseUnit(
            dataset="cail",
            split="train",
            case_id="c1",
            group_id="c1",
            target_defendant="段某",
            fact_raw="事实",
            charges=("故意伤害",),
            penalty_type=penalty,
            imprisonment_months=12,
            source_path="data.json:1",
        )


def test_domain_json_round_trip() -> None:
    factors = LegalFactors(
        amount=8000.0,
        surrender=True,
        restitution=False,
        confession=True,
        role="principal",
    )
    spec = InterventionSpec(
        parent_case_id="c1",
        target_defendant="段某",
        intervention_type="sentence_rank",
        source_factors=factors,
        target_factors=LegalFactors(**{**factors.to_dict(), "restitution": True}),
        target_charge=None,
        rank_direction=-1,
        changed_fields=("restitution",),
        rule_id="restitution_mitigates",
    )
    assert InterventionSpec.from_dict(spec.to_dict()) == spec


def test_config_precedence_cli_over_environment_over_yaml(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("data:\n  root: /yaml\ntraining:\n  seed: 42\n", encoding="utf-8")
    config = load_config(
        path,
        env={"LEGAL_LANDSCAPE_DATA__ROOT": "/env"},
        overrides={"data.root": "/cli"},
    )
    assert config["data"]["root"] == "/cli"
    assert config["training"]["seed"] == 42
