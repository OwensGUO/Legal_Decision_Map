"""Automatic validation for generated legal counterfactuals."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any

from legal_landscape.data.sanitize import MASK, sanitize_text
from legal_landscape.factors.extract import (
    RESTITUTION_CUES,
    conduct_cues,
    cue_polarity,
    extract_factors,
)
from legal_landscape.factors.schema import InterventionSpec

_PENALTY = re.compile(
    r"有期徒刑|拘役|管制(?!刀具)|缓刑|罚金|死刑|无期|从轻|减轻|从重|免予刑事处罚|判处|刑期"
)
_AMOUNT = re.compile(r"([0-9][0-9,，]*(?:\.[0-9]+)?)\s*(万)?\s*元")
_WHITESPACE = re.compile(r"\s+")
_NGRAM = 8
# Errors meaning the text reveals labels or outcomes; these are never used for training,
# including in the unfiltered ablation.
LEAKAGE_ERRORS = frozenset(
    {
        "label_leakage",
        "target_label_leakage",
        "article_leakage",
        "conclusion_leakage",
        "sentence_leakage",
        "mask_restored",
    }
)
_LEAKAGE_BY_CATEGORY = {
    "charge": "label_leakage",
    "article": "article_leakage",
    "conclusion": "conclusion_leakage",
    "result_statement": "conclusion_leakage",
}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]
    counterfactual_text: str | None
    parsed: dict[str, Any] | None
    sanitized_text: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def has_leakage(errors: Iterable[str]) -> bool:
    return any(error in LEAKAGE_ERRORS for error in errors)


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


def _clauses(text: str) -> list[str]:
    return [item for item in re.findall(r"[^。；！？\n]+", text) if item.strip()]


def _other_defendants(spec: InterventionSpec) -> tuple[str, ...]:
    return tuple(
        name for name in spec.co_defendants if name and name != spec.target_defendant
    )


def _names_overlap(target: str, others: tuple[str, ...]) -> bool:
    names = (target, *others)
    return any(
        left != right and (left in right or right in left)
        for left in names
        for right in others
    )


def _sentence_supports_change(sentence: str, spec: InterventionSpec) -> bool:
    for name in spec.changed_fields:
        value = getattr(spec.target_factors, name)
        if name == "restitution" and value is not None:
            if cue_polarity(sentence, RESTITUTION_CUES) != value:
                return False
        elif name == "conduct" and isinstance(value, str):
            if value not in conduct_cues(sentence):
                return False
        elif name == "amount" and value is not None and not _amount_appears(sentence, float(value)):
            return False
    return True


def _target_scope(text: str, spec: InterventionSpec) -> str:
    """Sentences that name the target and do not name another defendant."""
    others = _other_defendants(spec)
    if not others:
        return text
    return "".join(
        sentence
        for sentence in _clauses(text)
        if spec.target_defendant in sentence
        and not any(name in sentence for name in others)
    )


def _attribution_ambiguous(text: str, spec: InterventionSpec) -> bool:
    """Shared clauses, bare pronouns, or overlapping names cannot prove a defendant edit."""
    others = _other_defendants(spec)
    if not others:
        return False
    if _names_overlap(spec.target_defendant, others):
        return True
    for sentence in _clauses(text):
        if not _sentence_supports_change(sentence, spec):
            continue
        mentions_target = spec.target_defendant in sentence
        mentions_other = any(name in sentence for name in others)
        if mentions_target == mentions_other:
            return True
    return False


def _target_change_appears(text: str, spec: InterventionSpec) -> bool:
    text = _target_scope(text, spec)
    for name in spec.changed_fields:
        value = getattr(spec.target_factors, name)
        if name == "restitution" and value is not None:
            if cue_polarity(text, RESTITUTION_CUES) is not value:
                return False
        if name == "conduct" and isinstance(value, str):
            cues = conduct_cues(text)
            source = spec.source_factors.conduct
            if value not in cues or (source is not None and source != value and source in cues):
                return False
        if name == "amount" and value is not None and not _amount_appears(text, float(value)):
            return False
        source_amount = spec.source_factors.amount
        if (
            name == "amount"
            and source_amount is not None
            and value is not None
            and abs(float(source_amount) - float(value)) > max(1.0, abs(float(value)) * 1e-3)
            and _amount_appears(text, float(source_amount))
        ):
            return False
    return True


def _factor_drift(parent_text: str, text: str, spec: InterventionSpec) -> list[str]:
    """Re-extract factors from both texts instead of trusting the generator's report."""
    others = _other_defendants(spec)
    before = extract_factors(
        parent_text, target_defendant=spec.target_defendant, co_defendants=others
    )
    after = extract_factors(text, target_defendant=spec.target_defendant, co_defendants=others)
    errors: list[str] = []
    for name in before.to_dict():
        if name in spec.changed_fields:
            if name == "restitution" and after.restitution != spec.target_factors.restitution:
                errors.append("factor_mismatch:restitution")
        elif getattr(after, name) != getattr(before, name):
            errors.append(f"non_target_changed:{name}")
    for other in others:
        company = tuple(name for name in (spec.target_defendant, *others) if name != other)
        parent_other = extract_factors(
            parent_text, target_defendant=other, co_defendants=company
        )
        child_other = extract_factors(text, target_defendant=other, co_defendants=company)
        if parent_other != child_other:
            errors.append(f"non_target_defendant_changed:{other}")
    return errors


def _leakage(
    parent_text: str, text: str, spec: InterventionSpec, lexicon: tuple[str, ...]
) -> tuple[list[str], str]:
    """Count-based checks: the generator may not add what the parent text does not contain."""
    errors: list[str] = []
    parent = sanitize_text(parent_text, charge_lexicon=lexicon)
    generated = sanitize_text(text, charge_lexicon=lexicon)
    before = Counter(item.category for item in parent.redactions)
    after = Counter(item.category for item in generated.redactions)
    for category, error in _LEAKAGE_BY_CATEGORY.items():
        if after[category] > before[category]:
            errors.append(error)
    if spec.target_charge:
        label = f"{spec.target_charge}罪"
        if text.count(label) > parent_text.count(label):
            errors.append("target_label_leakage")
    if len(_PENALTY.findall(text)) > len(_PENALTY.findall(parent_text)):
        errors.append("sentence_leakage")
    if text.count("××") < parent_text.count("××") or text.count(MASK) < parent_text.count(MASK):
        errors.append("mask_restored")
    return errors, generated.strict


def _novel_ngram_ratio(parent_text: str, text: str) -> float:
    grams = {text[index : index + _NGRAM] for index in range(len(text) - _NGRAM + 1)}
    if not grams:
        return 0.0
    known = {parent_text[index : index + _NGRAM] for index in range(len(parent_text) - _NGRAM + 1)}
    return len(grams - known) / len(grams)


def _normalized(text: str) -> str:
    return _WHITESPACE.sub("", text)


def validate_generation(
    parent_text: str,
    spec: InterventionSpec,
    raw_response: str,
    *,
    min_similarity: float = 0.5,
    charge_lexicon: Iterable[str] = (),
) -> ValidationResult:
    """Check one realized intervention.

    A faithful rewrite keeps most of the parent text, so the text must differ from the parent
    but stay at least ``min_similarity`` similar to it. Label, article, conclusion, and
    sentence mentions only count when the generator added them beyond the parent text.
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
    reported = parsed.get("target_defendant", spec.target_defendant)
    if reported != spec.target_defendant or spec.target_defendant not in text:
        errors.append("identity_not_preserved")
    if _attribution_ambiguous(text, spec):
        errors.append("attribution_ambiguous")
    elif not _target_change_appears(text, spec):
        errors.append("target_change_missing")
    errors.extend(_factor_drift(parent_text, text, spec))
    leakage, sanitized = _leakage(parent_text, text, spec, tuple(charge_lexicon))
    errors.extend(leakage)
    similarity = SequenceMatcher(None, parent_text, text, autojunk=False).ratio()
    if _normalized(text) == _normalized(parent_text):
        errors.append("unchanged")
    elif similarity < min_similarity:
        errors.append("excessive_rewrite")
    metrics = {
        "similarity": similarity,
        "novel_ngram_ratio": _novel_ngram_ratio(parent_text, text),
    }
    return ValidationResult(
        not errors, tuple(dict.fromkeys(errors)), text, parsed, sanitized, metrics
    )
