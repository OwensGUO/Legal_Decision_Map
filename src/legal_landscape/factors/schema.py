"""Structured legal factors and counterfactual interventions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Role = Literal["principal", "accessory", "unknown"]
InterventionType = Literal["invariant", "charge_flip", "sentence_rank"]


@dataclass(frozen=True)
class LegalFactors:
    amount: float | None = None
    surrender: bool | None = None
    restitution: bool | None = None
    confession: bool | None = None
    role: Role = "unknown"
    conduct: str | None = None

    def __post_init__(self) -> None:
        if self.amount is not None and self.amount < 0:
            raise ValueError("amount must be non-negative")
        if self.role not in {"principal", "accessory", "unknown"}:
            raise ValueError(f"invalid role: {self.role}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LegalFactors:
        return cls(**value)


@dataclass(frozen=True)
class InterventionSpec:
    parent_case_id: str
    target_defendant: str
    intervention_type: InterventionType
    source_factors: LegalFactors
    target_factors: LegalFactors
    target_charge: str | None
    rank_direction: Literal[-1, 0, 1] | None
    changed_fields: tuple[str, ...]
    rule_id: str

    def __post_init__(self) -> None:
        if not self.parent_case_id or not self.target_defendant or not self.rule_id:
            raise ValueError("parent, defendant, and rule_id are required")
        if not self.changed_fields:
            raise ValueError("changed_fields cannot be empty")
        allowed = set(LegalFactors.__dataclass_fields__)
        if not set(self.changed_fields).issubset(allowed):
            raise ValueError("changed_fields contains an unknown factor")
        if self.intervention_type == "charge_flip" and not self.target_charge:
            raise ValueError("charge_flip requires target_charge")
        if self.intervention_type == "sentence_rank" and self.rank_direction not in {-1, 1}:
            raise ValueError("sentence_rank requires rank_direction -1 or 1")
        if self.intervention_type != "sentence_rank" and self.rank_direction is not None:
            raise ValueError("rank_direction is only valid for sentence_rank")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["changed_fields"] = list(self.changed_fields)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InterventionSpec:
        return cls(
            **{
                **value,
                "source_factors": LegalFactors.from_dict(value["source_factors"]),
                "target_factors": LegalFactors.from_dict(value["target_factors"]),
                "changed_fields": tuple(value["changed_fields"]),
            }
        )
