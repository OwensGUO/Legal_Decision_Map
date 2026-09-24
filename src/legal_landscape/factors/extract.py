"""Deterministic seed extraction for supported legal factors."""

from __future__ import annotations

import re

from legal_landscape.factors.schema import LegalFactors

_AMOUNT = re.compile(
    r"(?:涉案(?:金额)?|价值|获利|金额)[^，。；]{0,12}?([0-9]+(?:\.[0-9]+)?)\s*(万)?元"
)


def _flag(text: str, positives: tuple[str, ...], negatives: tuple[str, ...]) -> bool | None:
    if any(item in text for item in negatives):
        return False
    if any(item in text for item in positives):
        return True
    return None


def _conduct(text: str) -> str | None:
    patterns = (
        ("秘密窃取", ("秘密窃取", "盗走", "偷走")),
        ("虚构事实使被害人自愿交付", ("虚构事实", "骗取", "诈骗")),
        ("组织并经营赌博场所", ("开设赌场", "组织赌博", "抽水获利")),
        ("参与赌博", ("参与赌博", "聚众赌博")),
        ("暴力胁迫取财", ("暴力抢劫", "持刀抢劫", "胁迫交付")),
        ("趁人不备公开夺取", ("趁人不备", "抢夺")),
        ("无事生非随意殴打", ("无事生非", "随意殴打", "寻衅滋事")),
        ("因个人纠纷定向伤害", ("个人纠纷", "故意伤害", "致人轻伤")),
    )
    return next(
        (normalized for normalized, cues in patterns if any(cue in text for cue in cues)), None
    )


def extract_factors(text: str, *, target_defendant: str) -> LegalFactors:
    """Extract auditable weak labels; unknown values stay ``None``."""
    del target_defendant  # Reserved for a future defendant-local sentence segmenter.
    amount_match = _AMOUNT.search(text)
    amount = None
    if amount_match:
        amount = float(amount_match.group(1)) * (10000.0 if amount_match.group(2) else 1.0)
    role = "unknown"
    if "从犯" in text:
        role = "accessory"
    elif "主犯" in text:
        role = "principal"
    return LegalFactors(
        amount=amount,
        surrender=_flag(text, ("投案自首", "主动投案", "构成自首"), ("不构成自首", "无自首")),
        restitution=_flag(text, ("退赃", "退赔", "返还赃款"), ("未退赃", "拒不退赔")),
        confession=_flag(text, ("认罪", "如实供述", "坦白"), ("拒不认罪", "不认罪")),
        role=role,
        conduct=_conduct(text),
    )
