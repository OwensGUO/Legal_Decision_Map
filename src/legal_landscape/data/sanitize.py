"""Auditable conservative and strict legal-text sanitization."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Redaction:
    category: str
    text: str
    start: int
    end: int
    mode: str


@dataclass(frozen=True)
class SanitizedText:
    raw: str
    conservative: str
    strict: str
    redactions: tuple[Redaction, ...]


def _redact(
    text: str, patterns: list[tuple[str, re.Pattern[str]]], mode: str
) -> tuple[str, list[Redaction]]:
    found: list[Redaction] = []
    for category, pattern in patterns:
        for match in pattern.finditer(text):
            found.append(Redaction(category, match.group(0), match.start(), match.end(), mode))
    result = text
    for item in sorted(found, key=lambda value: value.start, reverse=True):
        result = result[: item.start] + "[已屏蔽]" + result[item.end :]
    return result, found


def sanitize_text(text: str, *, charges: tuple[str, ...] = ()) -> SanitizedText:
    """Return conservative and strict variants while retaining an audit trail."""
    conservative_patterns = [
        ("result_statement", re.compile(r"(?:公诉机关指控|本院认为)")),
        (
            "sentencing_recommendation",
            re.compile(r"(?:建议判处|量刑建议)[^。；]*[。；]?"),
        ),
    ]
    conservative, first = _redact(text, conservative_patterns, "conservative")
    strict_patterns: list[tuple[str, re.Pattern[str]]] = [
        (
            "article",
            re.compile(r"(?:《中华人民共和国刑法》|刑法)?第[一二三四五六七八九十百千零〇0-9]+条"),
        )
    ]
    for charge in sorted(set(charges), key=len, reverse=True):
        strict_patterns.append(("charge", re.compile(re.escape(charge) + "罪?")))
    strict, second = _redact(conservative, strict_patterns, "strict")
    return SanitizedText(text, conservative, strict, tuple(first + second))
