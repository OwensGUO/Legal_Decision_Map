"""Validation helpers for extracted or generated legal factors."""

from __future__ import annotations

from legal_landscape.factors.schema import LegalFactors


def validate_factors(factors: LegalFactors) -> tuple[str, ...]:
    errors: list[str] = []
    if factors.amount is not None and (factors.amount < 0 or factors.amount > 10**12):
        errors.append("amount_out_of_range")
    if factors.role not in {"principal", "accessory", "unknown"}:
        errors.append("invalid_role")
    if factors.conduct is not None and not factors.conduct.strip():
        errors.append("empty_conduct")
    return tuple(errors)
