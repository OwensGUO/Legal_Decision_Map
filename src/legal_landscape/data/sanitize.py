"""Auditable, label-agnostic conservative and strict legal-text sanitization."""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

SANITIZER_VERSION = "v2"
MASK = "[已屏蔽]"


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


_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_SEGMENT = re.compile(r"[^。；！？\n]+")
_RESULT_STATEMENT = re.compile(r"(?:公诉机关|人民检察院|检察院)指控")
# Each marker opens the prosecution's or court's legal conclusion; the rest of the
# sentence carries charge names, cited provisions, or a sentencing recommendation.
_CONCLUSION = re.compile(
    r"(?:公诉机关|检察院|检察机关|公诉人)认为|本院认为|据此|综上"
    r"|其(?:上述)?行为已?(?:触犯|构成)|(?:应当|应)以|追究其?刑事责任"
    r"|提请(?:本院|法院|依法)|依法(?:惩处|判处|处罚|追究)|量刑建议"
    r"|建议(?=[^。；！？\n]{0,40}?(?:判|量刑|幅度|刑|罪|撤销|并罚|拘役|管制|缓刑|罚金"
    r"|[一二三四五六七八九十两0-9]+(?:年|个月)))"
    r"|(?:依据|依照|根据)《"
)
_NUMBER = "[一二三四五六七八九十百千零〇两0-9]+"
_ARTICLE = re.compile(
    rf"(?:《[^》]{{0,30}}》|刑法)?第{_NUMBER}条(?:之{_NUMBER})?(?:第{_NUMBER}款)?"
    rf"(?:第{_NUMBER}项)?|第{_NUMBER}款"
)
# Charge names after a legal cue, including names absent from the training vocabulary.
_CUED_CHARGE = re.compile(
    r"(?:犯|以|构成|涉嫌|触犯)([\u4e00-\u9fa5、]{2,12}?(?<![认悔无有获得])罪)"
)


def expand_charge_label(label: str) -> tuple[str, ...]:
    """Surface forms of a label such as ``[走私、贩卖、运输、制造]毒品``."""
    value = label.strip().removesuffix("罪")
    parts: list[list[str]] = []
    position = 0
    for match in _BRACKET.finditer(value):
        parts.append([value[position : match.start()]])
        options = [item for item in match.group(1).split("、") if item]
        parts.append([*options, "、".join(options)])
        position = match.end()
    parts.append([value[position:]])
    return tuple(dict.fromkeys("".join(combination) for combination in itertools.product(*parts)))


def charge_lexicon(labels: Iterable[str]) -> tuple[str, ...]:
    """Longest-first surface forms of every label, independent of any single case."""
    names = {name for label in labels for name in expand_charge_label(str(label)) if len(name) >= 2}
    return tuple(sorted(names, key=lambda item: (-len(item), item)))


@lru_cache(maxsize=8)
def _lexicon_pattern(lexicon: tuple[str, ...]) -> re.Pattern[str] | None:
    if not lexicon:
        return None
    names = "(?:" + "|".join(re.escape(name) for name in lexicon) + ")"
    # Charge mentions, and indictment section headings such as "一、强迫交易" before a delimiter.
    return re.compile(
        rf"{names}(?:犯罪|罪|案)|(?<=[一二三四五六七八九十]、){names}(?=[；：:，。\n])"
    )


def _conclusion_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for segment in _SEGMENT.finditer(text):
        marker = _CONCLUSION.search(segment.group(0))
        if marker is not None:
            spans.append(("conclusion", segment.start() + marker.start(), segment.end()))
    return spans


def _apply(
    text: str, spans: list[tuple[str, int, int]], mode: str
) -> tuple[str, list[Redaction]]:
    found = [
        Redaction(category, text[start:end], start, end, mode) for category, start, end in spans
    ]
    merged: list[list[int]] = []
    for _, start, end in sorted(spans, key=lambda item: (item[1], item[2])):
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    result = text
    for start, end in reversed(merged):
        result = result[:start] + MASK + result[end:]
    return result, found


def sanitize_text(text: str, *, charge_lexicon: Iterable[str] = ()) -> SanitizedText:
    """Return conservative and strict variants while retaining an audit trail.

    The same lexicon must be used for every case and split: masking only a case's own
    gold labels would make the model input depend on the label.
    """
    first_spans = [
        ("result_statement", match.start(), match.end())
        for match in _RESULT_STATEMENT.finditer(text)
    ]
    first_spans.extend(_conclusion_spans(text))
    conservative, first = _apply(text, first_spans, "conservative")

    second_spans = [
        ("article", match.start(), match.end()) for match in _ARTICLE.finditer(conservative)
    ]
    pattern = _lexicon_pattern(tuple(charge_lexicon))
    if pattern is not None:
        second_spans.extend(
            ("charge", match.start(), match.end()) for match in pattern.finditer(conservative)
        )
    second_spans.extend(
        ("charge", match.start(1), match.end(1)) for match in _CUED_CHARGE.finditer(conservative)
    )
    strict, second = _apply(conservative, second_spans, "strict")
    return SanitizedText(text, conservative, strict, tuple(first + second))
