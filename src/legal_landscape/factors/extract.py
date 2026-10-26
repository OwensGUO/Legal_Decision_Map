"""Deterministic seed extraction for supported legal factors."""

from __future__ import annotations

import re
from collections.abc import Iterable

from legal_landscape.factors.schema import LegalFactors

_AMOUNT = re.compile(
    r"(?:涉案(?:金额)?|价值|获利|金额)[^，。；]{0,12}?([0-9]+(?:\.[0-9]+)?)\s*(万)?元"
)
_SENTENCE = re.compile(r"[^。；！？\n]*[。；！？\n]?")

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


def _flag(text: str, positives: tuple[str, ...], negatives: tuple[str, ...]) -> bool | None:
    if any(item in text for item in negatives):
        return False
    if any(item in text for item in positives):
        return True
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
        amount = float(amount_match.group(1)) * (10000.0 if amount_match.group(2) else 1.0)
    role = "unknown"
    if "从犯" in local:
        role = "accessory"
    elif "主犯" in local:
        role = "principal"
    return LegalFactors(
        amount=amount,
        surrender=_flag(local, ("投案自首", "主动投案", "构成自首"), ("不构成自首", "无自首")),
        restitution=_flag(local, ("退赃", "退赔", "返还赃款"), ("未退赃", "拒不退赔")),
        confession=_flag(local, ("认罪", "如实供述", "坦白"), ("拒不认罪", "不认罪")),
        role=role,
        conduct=_conduct(local),
    )
