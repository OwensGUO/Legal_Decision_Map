"""Rules that alter structured factors before language realization."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from legal_landscape.factors.schema import InterventionSpec, LegalFactors

_TARGET_CONDUCT = {
    "盗窃": "秘密窃取",
    "诈骗": "虚构事实使被害人自愿交付",
    "赌博": "参与赌博",
    "开设赌场": "组织并经营赌博场所",
    "故意伤害": "因个人纠纷定向伤害",
    "寻衅滋事": "无事生非随意殴打",
    "抢夺": "趁人不备公开夺取",
    "抢劫": "暴力胁迫取财",
}


def _flip_targets(charge: str, boundaries: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    """Every distinct neighbour of ``charge`` across all configured boundary groups."""
    targets: dict[str, None] = {}
    for group in boundaries:
        if charge in group:
            targets.update((candidate, None) for candidate in group if candidate != charge)
    return tuple(targets)


def propose_interventions(
    *,
    parent_case_id: str,
    defendant: str,
    charge: str,
    factors: LegalFactors,
    charge_boundaries: Iterable[tuple[str, ...]],
) -> tuple[InterventionSpec, ...]:
    """Create conditional soft-constraint proposals from a source case."""
    proposals: list[InterventionSpec] = []
    if factors.amount is not None:
        proposals.append(
            InterventionSpec(
                parent_case_id=parent_case_id,
                target_defendant=defendant,
                intervention_type="invariant",
                source_factors=factors,
                target_factors=replace(factors, amount=round(factors.amount * 1.01, 2)),
                target_charge=None,
                rank_direction=None,
                changed_fields=("amount",),
                rule_id=f"{charge}:amount_same_legal_band",
            )
        )
    for target_charge in _flip_targets(charge, charge_boundaries):
        proposals.append(
            InterventionSpec(
                parent_case_id=parent_case_id,
                target_defendant=defendant,
                intervention_type="charge_flip",
                source_factors=factors,
                target_factors=replace(factors, conduct=_TARGET_CONDUCT[target_charge]),
                target_charge=target_charge,
                rank_direction=None,
                changed_fields=("conduct",),
                rule_id=f"boundary:{charge}->{target_charge}",
            )
        )
    removing = factors.restitution is True
    proposals.append(
        InterventionSpec(
            parent_case_id=parent_case_id,
            target_defendant=defendant,
            intervention_type="sentence_rank",
            source_factors=factors,
            target_factors=replace(factors, restitution=not removing),
            target_charge=None,
            rank_direction=1 if removing else -1,
            changed_fields=("restitution",),
            rule_id=f"{charge}:restitution_conditional_direction",
        )
    )
    return tuple(proposals)
