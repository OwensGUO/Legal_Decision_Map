"""Versioned prompts for constrained counterfactual realization."""

from __future__ import annotations

import json

from legal_landscape.factors.schema import InterventionSpec

PROMPT_VERSION = "v2"

_FIELD_INSTRUCTIONS = {
    "amount": "把涉案金额改写为 realized_factors.amount 给出的数值，用阿拉伯数字加“元”写出。",
    "restitution": (
        "realized_factors.restitution 为 true 时写明被告人已退赃或退赔；"
        "为 false 时写明“未退赃”或“未退赔”。"
    ),
    "conduct": (
        "把被告人的行为方式改写为 realized_factors.conduct 所描述的方式，"
        "并在文中原样写出该短语。"
    ),
}


def build_messages(parent_text: str, spec: InterventionSpec) -> list[dict[str, str]]:
    system = (
        "你是法律事实改写器。只能实现给定结构化干预，不得自行决定罪名、法律标签或精确刑期。"
        "counterfactual_text 必须是改写后的完整案件事实：只修改干预涉及的内容，"
        "其余文字尽量逐字保留，"
        "保持未修改因素与目标被告身份不变。不得写出任何罪名（例如“××罪”）或刑罚。"
        "不要输出思维过程。只返回JSON。"
    )
    user = {
        "source_text": parent_text,
        "intervention": spec.to_dict(),
        "rewrite_rules": [
            _FIELD_INSTRUCTIONS.get(field, f"按 realized_factors.{field} 改写相应事实。")
            for field in spec.changed_fields
        ],
        "response_schema": {
            "counterfactual_text": "string",
            "target_defendant": spec.target_defendant,
            "changed_fields": list(spec.changed_fields),
            "realized_factors": spec.target_factors.to_dict(),
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]
