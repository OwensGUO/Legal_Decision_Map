"""Versioned prompts for constrained counterfactual realization."""

from __future__ import annotations

import json

from legal_landscape.factors.extract import CONDUCT_CUES
from legal_landscape.factors.schema import InterventionSpec

PROMPT_VERSION = "v3"

_CUES = dict(CONDUCT_CUES)


def _amount(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _edit(field: str, value: object, defendant: str) -> str:
    """Natural-language edit that names facts only, never a charge, rule, or direction."""
    if field == "amount" and isinstance(value, int | float):
        return f"把{defendant}的涉案金额改写为{_amount(float(value))}元，用阿拉伯数字加“元”写出。"
    if field == "restitution":
        if value:
            return f"写明{defendant}案发后已退赃或退赔，并删除原文中未退赃、未退赔的表述。"
        return f"写明{defendant}未退赃或未退赔，并删除原文中已退赃、退赔的表述。"
    if field == "conduct" and isinstance(value, str):
        cues = "、".join(_CUES.get(value, ()))
        return (
            f"把{defendant}的行为方式改写为“{value}”的具体事实，使用以下表述之一：{cues}；"
            "同时删除或改写原文中与之矛盾的行为描述。"
        )
    return f"把{defendant}的{field}改写为{value}。"


def build_messages(parent_text: str, spec: InterventionSpec) -> list[dict[str, str]]:
    system = (
        "你是法律事实改写器。只修改指定的事实要素，其余文字尽量逐字保留，"
        "保持其他事实要素与被告人身份不变。原文中的“××”和“[已屏蔽]”是遮蔽标记，"
        "必须原样保留，不得补全、替换或删除。不得写出任何罪名、法条、刑罚、量刑建议"
        "或法律评价（例如“构成××罪”“从轻处罚”）。不要输出思维过程。"
        "只返回JSON对象：{\"counterfactual_text\": 改写后的完整案件事实}。"
    )
    user = {
        "source_text": parent_text,
        "target_defendant": spec.target_defendant,
        "edits": [
            _edit(field, getattr(spec.target_factors, field), spec.target_defendant)
            for field in spec.changed_fields
        ],
        "response_schema": {"counterfactual_text": "string"},
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]
