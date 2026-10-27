"""Auditable, label-agnostic conservative and strict legal-text sanitization."""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

SANITIZER_VERSION = "v3"
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
# Markers that introduce the prosecution's or court's conclusion for the current case.
# Boundaries keep them from firing inside 根据此前 / 答应以.
_STRONG_CONCLUSION = re.compile(
    r"(?:公诉机关|检察院|检察机关|公诉人)认为|本院认为|(?<!根)据此|综上"
    r"|追究其?刑事责任"
    r"|提请(?:本院|法院|依法)|量刑建议"
    r"|建议(?=[^。；！？\n]{0,40}?(?:判|量刑|幅度|刑|罪|撤销|并罚|拘役|管制|缓刑|罚金"
    r"|[一二三四五六七八九十两0-9]+(?:年|个月)))"
    r"|(?:依据|依照|根据)《"
    r"|(?:应当以|(?<!答)应以)"
)
# Weaker sentencing language. A historical cue in the same clause (曾、此前、前科)
# keeps the past punishment instead of treating it as this case's conclusion.
_WEAK_CONCLUSION = re.compile(
    r"其(?:上述)?行为已?(?:触犯|构成)"
    r"|依法(?:惩处|判处|处罚|追究)"
    r"|(?<!不)(?:可以|应当|应|酌情)(?:依法)?"
    r"(?:从轻|从重|减轻|加重|从宽|从严)"
    r"(?:、(?:从轻|从重|减轻|加重|从宽|从严))*处罚"
)
_HISTORY = re.compile(r"曾|曾经|以前|此前|当年|前科")
_NUMBER = "[一二三四五六七八九十百千零〇两0-9]+"
_ARTICLE = re.compile(
    rf"(?:《[^》]{{0,30}}》|刑法)?第{_NUMBER}条(?:之{_NUMBER})?(?:第{_NUMBER}款)?"
    rf"(?:第{_NUMBER}项)?|第{_NUMBER}款"
)
# Charge names after a legal cue. Bare 以 is not a cue: it also occurs in 不以 and 答应以.
_CUED_CHARGE = re.compile(
    r"(?:犯|构成|涉嫌|触犯|(?:应当|(?<!答)应)以)"
    r"([\u4e00-\u9fa5、]{2,12}?(?<![认悔无有获得])罪)"
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


def _locally_historical(segment: str, marker_start: int) -> bool:
    """True when a past-punishment cue governs this marker inside the current clause."""
    local = segment[:marker_start]
    cut = max(local.rfind(separator) for separator in ("，", ",", "、"))
    return _HISTORY.search(local[cut + 1 :]) is not None


def _conclusion_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for segment in _SEGMENT.finditer(text):
        body = segment.group(0)
        starts = [match.start() for match in _STRONG_CONCLUSION.finditer(body)]
        starts.extend(
            match.start()
            for match in _WEAK_CONCLUSION.finditer(body)
            if not _locally_historical(body, match.start())
        )
        if starts:
            spans.append(("conclusion", segment.start() + min(starts), segment.end()))
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
