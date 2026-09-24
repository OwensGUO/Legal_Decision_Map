"""Automatic validation for generated legal counterfactuals."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

from legal_landscape.factors.schema import InterventionSpec

_SENTENCE_LEAKAGE = re.compile(r"(?:判处|刑期)[^。；]{0,20}?[0-9一二三四五六七八九十]+(?:个?月|年)")
_AMOUNT = re.compile(r"([0-9][0-9,，]*(?:\.[0-9]+)?)\s*(万)?\s*元")
_RESTITUTION_DONE = re.compile(r"(?<!未)(?<!未能)(?<!拒不)(?<!没有)(?:退赃|退赔|返还赃款)")
_RESTITUTION_NOT_DONE = (
    "未退赃",
    "未退赔",
    "未能退赃",
    "未能退赔",
    "拒不退赔",
    "没有退赃",
    "没有退赔",
)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]
    counterfactual_text: str | None
    parsed: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _factor_checks(parsed: dict[str, Any], spec: InterventionSpec) -> list[str]:
    errors: list[str] = []
    realized = parsed.get("realized_factors")
    if not isinstance(realized, dict):
        return ["missing_realized_factors"]
    source = spec.source_factors.to_dict()
    target = spec.target_factors.to_dict()
    for field, expected in target.items():
        if realized.get(field) != expected:
            errors.append(f"factor_mismatch:{field}")
        if field not in spec.changed_fields and realized.get(field) != source.get(field):
            errors.append(f"non_target_changed:{field}")
    if tuple(parsed.get("changed_fields") or ()) != spec.changed_fields:
        errors.append("changed_fields_mismatch")
    return errors


def mentioned_amounts(text: str) -> list[float]:
    """Yuan amounts written as ``12468.45元``, ``12,468元`` or ``1.25万元``."""
    amounts: list[float] = []
    for match in _AMOUNT.finditer(text):
        try:
            value = float(re.sub("[,，]", "", match.group(1)))
        except ValueError:
            continue
        amounts.append(value * 10000.0 if match.group(2) else value)
    return amounts


def _amount_appears(text: str, amount: float) -> bool:
    # The invariant rule moves amounts by 1%, so the tolerance must stay well below that.
    tolerance = max(1.0, amount * 1e-3)
    return any(abs(value - amount) <= tolerance for value in mentioned_amounts(text))


def _target_change_appears(text: str, spec: InterventionSpec) -> bool:
    for field in spec.changed_fields:
        value = getattr(spec.target_factors, field)
        if field == "restitution" and value is True and not _RESTITUTION_DONE.search(text):
            return False
        if (
            field == "restitution"
            and value is False
            and not any(cue in text for cue in _RESTITUTION_NOT_DONE)
        ):
            return False
        if field == "conduct" and isinstance(value, str) and value not in text:
            return False
        if field == "amount" and value is not None and not _amount_appears(text, value):
            return False
    return True


def _normalized(text: str) -> str:
    return _WHITESPACE.sub("", text)


def validate_generation(
    parent_text: str,
    spec: InterventionSpec,
    raw_response: str,
    *,
    min_similarity: float = 0.5,
) -> ValidationResult:
    """Check one realized intervention.

    A faithful rewrite keeps most of the parent text, so the text must differ from the parent
    but stay at least ``min_similarity`` similar to it. Label and sentence leakage only count
    when the generator introduced them; source facts often mention the charge under
    investigation, which is outside the generator's control.
    """
    errors: list[str] = []
    try:
        parsed = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        return ValidationResult(False, ("invalid_json",), None, None)
    if not isinstance(parsed, dict):
        return ValidationResult(False, ("invalid_json_object",), None, None)
    text = parsed.get("counterfactual_text")
    if not isinstance(text, str) or not text.strip():
        return ValidationResult(False, ("missing_counterfactual_text",), None, parsed)
    if parsed.get("target_defendant") != spec.target_defendant or spec.target_defendant not in text:
        errors.append("identity_not_preserved")
    errors.extend(_factor_checks(parsed, spec))
    if not _target_change_appears(text, spec):
        errors.append("target_change_missing")
    if spec.target_charge:
        label = f"{spec.target_charge}罪"
        if label in text and label not in parent_text:
            errors.append("target_label_leakage")
    if _SENTENCE_LEAKAGE.search(text) and not _SENTENCE_LEAKAGE.search(parent_text):
        errors.append("sentence_leakage")
    if _normalized(text) == _normalized(parent_text):
        errors.append("unchanged")
    elif SequenceMatcher(None, parent_text, text, autojunk=False).ratio() < min_similarity:
        errors.append("excessive_rewrite")
    return ValidationResult(not errors, tuple(dict.fromkeys(errors)), text, parsed)
