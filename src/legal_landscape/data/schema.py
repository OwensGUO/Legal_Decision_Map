"""Normalized immutable case schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class PenaltyType(StrEnum):
    FIXED_TERM = "fixed_term"
    LIFE = "life"
    DEATH = "death"
    DETENTION = "detention"
    CONTROL = "control"
    EXEMPT = "exempt"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CaseUnit:
    dataset: str
    split: str
    case_id: str
    group_id: str
    target_defendant: str
    fact_raw: str
    charges: tuple[str, ...]
    penalty_type: str
    imprisonment_months: int | None
    source_path: str

    def __post_init__(self) -> None:
        if not all((self.dataset, self.split, self.case_id, self.group_id, self.source_path)):
            raise ValueError("dataset, split, identifiers, and source_path are required")
        if not self.charges or any(not item.strip() for item in self.charges):
            raise ValueError("charges must contain non-empty labels")
        if self.penalty_type not in {item.value for item in PenaltyType}:
            raise ValueError(f"unknown penalty_type: {self.penalty_type}")
        if self.penalty_type != PenaltyType.FIXED_TERM and self.imprisonment_months is not None:
            raise ValueError("months are only valid for fixed-term imprisonment")
        if self.imprisonment_months is not None and self.imprisonment_months < 0:
            raise ValueError("months must be non-negative")

    @property
    def is_sentence_regression_eligible(self) -> bool:
        return (
            self.penalty_type == PenaltyType.FIXED_TERM
            and self.imprisonment_months is not None
            and len(self.charges) == 1
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["charges"] = list(self.charges)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CaseUnit:
        return cls(**{**value, "charges": tuple(value["charges"])})


def assert_group_split_integrity(cases: list[CaseUnit] | tuple[CaseUnit, ...]) -> None:
    """Reject any original case group appearing in more than one split."""
    seen: dict[str, str] = {}
    for case in cases:
        previous = seen.setdefault(case.group_id, case.split)
        if previous != case.split:
            raise ValueError(
                f"group {case.group_id!r} appears in multiple splits: {previous}, {case.split}"
            )
