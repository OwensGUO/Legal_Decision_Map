"""Rules that alter structured factors before language realization."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence
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

# Amount boundaries used by judicial interpretations for property and economic crimes
# ("数额较大/巨大/特别巨大"). An invariant edit must not move a case across one.
_AMOUNT_THRESHOLDS = (
    1000.0, 2000.0, 3000.0, 5000.0, 6000.0, 10000.0, 20000.0, 30000.0, 50000.0, 60000.0,
    100000.0, 200000.0, 300000.0, 400000.0, 500000.0, 1000000.0, 3000000.0, 5000000.0,
)


def _flip_targets(charge: str, boundaries: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    """Every distinct neighbour of ``charge`` across all configured boundary groups."""
    targets: dict[str, None] = {}
    for group in boundaries:
        if charge in group:
            targets.update((candidate, None) for candidate in group if candidate != charge)
    return tuple(targets)


def _same_band_amount(amount: float) -> float | None:
    """A 1% change that keeps the amount inside its statutory band, if one exists."""
    band = bisect_right(_AMOUNT_THRESHOLDS, amount)
    for factor in (1.01, 0.99):
        candidate = round(amount * factor, 2)
        if candidate != amount and bisect_right(_AMOUNT_THRESHOLDS, candidate) == band:
            return candidate
    return None


def propose_interventions(
    *,
    parent_case_id: str,
    defendant: str,
    charges: Sequence[str],
    factors: LegalFactors,
    charge_boundaries: Iterable[tuple[str, ...]],
) -> tuple[InterventionSpec, ...]:
    """Create conditional soft-constraint proposals from a source case."""
    if not charges:
        raise ValueError("interventions require at least one parent charge")
    label = "+".join(charges)
    proposals: list[InterventionSpec] = []
    amount = _same_band_amount(factors.amount) if factors.amount is not None else None
    if amount is not None:
        proposals.append(
            InterventionSpec(
                parent_case_id=parent_case_id,
                target_defendant=defendant,
                intervention_type="invariant",
                source_factors=factors,
                target_factors=replace(factors, amount=amount),
                target_charge=None,
                rank_direction=None,
                changed_fields=("amount",),
                rule_id=f"{label}:amount_same_legal_band",
            )
        )
    # A flip relabels the whole case with one charge, which is only sound for single-charge parents.
    flip_targets = _flip_targets(charges[0], charge_boundaries) if len(charges) == 1 else ()
    for target_charge in flip_targets:
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
                rule_id=f"boundary:{label}->{target_charge}",
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
            rule_id=f"{label}:restitution_conditional_direction",
        )
    )
    return tuple(proposals)
