"""Deterministic seed extraction for supported legal factors."""

from __future__ import annotations

import re
from collections.abc import Iterable

from legal_landscape.factors.schema import LegalFactors

_AMOUNT = re.compile(
    r"(?:涉案(?:金额)?|价值|获利|金额)[^0-9，。；]{0,12}?"
    r"([0-9][0-9,，]*(?:\.[0-9]+)?)\s*(万)?元"
)
_SENTENCE = re.compile(r"[^。；！？\n]*[。；！？\n]?")
_CLAUSE = re.compile(r"[^，。；！？\n]+")
_NEGATIONS = ("尚未", "未能", "拒不", "没有", "并未", "未", "不", "无")
# Words that may sit between a real negation and the behaviour it negates.
_NEGATION_GAPS = (
    "",
    "主动",
    "如实",
    "全部",
    "完全",
    "及时",
    "积极",
    "当庭",
    "彻底",
    "足额",
    "全额",
    "予以",
    "进行",
    "马上",
    "立即",
)
# Negation characters inside these words do not negate a later behaviour cue.
_ABSORBED_NEGATION = (
    "不但",
    "不仅",
    "不只",
    "不光",
    "不单",
    "不过",
    "不论",
    "不管",
    "未成年",
    "未满",
    "未婚",
    "无前科",
    "无犯罪前科",
)
SURRENDER_CUES = ("投案自首", "主动投案", "构成自首", "自首")
RESTITUTION_CUES = ("返还赃款", "退赃", "退赔")
CONFESSION_CUES = ("如实供述", "坦白", "认罪")

# Cues describe behaviour only. Charge names such as "诈骗" or "寻衅滋事" are excluded:
# they restate the label rather than the facts and would leak it through [FACTORS].
CONDUCT_CUES = (
    ("秘密窃取", ("秘密窃取", "盗走", "偷走", "窃得", "窃取", "扒窃", "盗取")),
    ("虚构事实使被害人自愿交付", ("虚构事实", "隐瞒真相", "骗取", "谎称", "冒充")),
    ("组织并经营赌博场所", ("组织赌博", "抽头渔利", "抽头", "抽水", "提供赌博场所")),
    ("参与赌博", ("参与赌博", "聚众赌博", "参赌")),
    ("暴力胁迫取财", ("持刀威胁", "劫取", "以暴力相威胁", "强行劫走", "胁迫交付")),
    ("趁人不备公开夺取", ("趁人不备", "乘人不备", "夺走", "夺取")),
    ("无事生非随意殴打", ("无事生非", "借故生非", "随意殴打")),
    (
        "因个人纠纷定向伤害",
        ("个人纠纷", "因琐事", "发生纠纷", "致人轻伤", "致其轻伤", "轻伤一级", "轻伤二级",
         "重伤一级", "重伤二级"),
    ),
)


def _marker_absorbed(prefix: str, marker_index: int) -> bool:
    """True when the negation character belongs to a different word, such as 不但 or 无前科."""
    for phrase in _ABSORBED_NEGATION:
        for offset in range(len(phrase)):
            start = marker_index - offset
            if start < 0:
                continue
            if prefix.startswith(phrase, start) and start + len(phrase) > marker_index:
                return True
    return False


def _cue_negated(clause: str, index: int) -> bool | None:
    """Whether the cue at ``index`` is negated.

    ``True`` means a negation attaches directly to this cue. ``False`` means the cue is
    affirmative. ``None`` means a nearby negation does not reliably govern the cue, so the
    caller must leave the factor unknown instead of emitting a definite label.
    """
    prefix = clause[max(0, index - 16) : index]
    for marker in sorted(_NEGATIONS, key=len, reverse=True):
        for gap in sorted(_NEGATION_GAPS, key=len, reverse=True):
            tail = marker + gap
            if not prefix.endswith(tail):
                continue
            if _marker_absorbed(prefix, len(prefix) - len(tail)):
                continue
            return True
    window_start = max(0, len(prefix) - 12)
    window = prefix[window_start:]
    for marker in _NEGATIONS:
        start = 0
        while True:
            found = window.find(marker, start)
            if found < 0:
                break
            if not _marker_absorbed(prefix, window_start + found):
                return None
            start = found + len(marker)
    return False


def cue_polarity(text: str, cues: tuple[str, ...]) -> bool | None:
    """Return True, False, or None when cues are absent or contradict each other.

    Longer cues win over suffixes such as 自首 inside 构成自首. Negation is not
    treated as the affirmative cue it contains, and a clause that both affirms
    and denies the same factor stays unknown.
    """
    ordered = tuple(sorted(set(cues), key=len, reverse=True))
    states: set[bool | None] = set()
    for clause in _CLAUSE.findall(text) or (text,):
        index = 0
        while index < len(clause):
            matched = next((cue for cue in ordered if clause.startswith(cue, index)), None)
            if matched is None:
                index += 1
                continue
            negated = _cue_negated(clause, index)
            states.add(None if negated is None else not negated)
            index += len(matched)
    if states == {True}:
        return True
    if states == {False}:
        return False
    return None


def conduct_cues(text: str) -> set[str]:
    """Every normalized conduct whose behaviour cues appear in ``text``."""
    return {normalized for normalized, cues in CONDUCT_CUES if any(cue in text for cue in cues)}


def _conduct(text: str) -> str | None:
    return next(
        (normalized for normalized, cues in CONDUCT_CUES if any(cue in text for cue in cues)),
        None,
    )


def _defendant_text(text: str, target: str, co_defendants: Iterable[str]) -> str:
    """Drop sentences that mention only other defendants of the same case."""
    others = [name for name in co_defendants if name and name != target]
    if not target or not others:
        return text
    containing_target = [name for name in others if target in name]
    kept: list[str] = []
    for sentence in _SENTENCE.findall(text):
        stripped = sentence
        for name in containing_target:
            stripped = stripped.replace(name, "")
        if target in stripped or not any(name in sentence for name in others):
            kept.append(sentence)
    return "".join(kept) or text


def extract_factors(
    text: str, *, target_defendant: str, co_defendants: Iterable[str] = ()
) -> LegalFactors:
    """Extract auditable weak labels; unknown values stay ``None``."""
    local = _defendant_text(text, target_defendant, co_defendants)
    amount_match = _AMOUNT.search(local)
    amount = None
    if amount_match:
        amount = float(re.sub("[,，]", "", amount_match.group(1))) * (
            10000.0 if amount_match.group(2) else 1.0
        )
    role = "unknown"
    if "从犯" in local:
        role = "accessory"
    elif "主犯" in local:
        role = "principal"
    return LegalFactors(
        amount=amount,
        surrender=cue_polarity(local, SURRENDER_CUES),
        restitution=cue_polarity(local, RESTITUTION_CUES),
        confession=cue_polarity(local, CONFESSION_CUES),
        role=role,
        conduct=_conduct(local),
    )
