"""Shared fixtures for counterfactual cache rows that training will accept."""

from __future__ import annotations


def cache_identity(parent_text: str, *, defendant: str = "某甲") -> dict[str, str]:
    from legal_landscape.counterfactual.generate import text_sha256
    from legal_landscape.counterfactual.prompts import PROMPT_VERSION
    from legal_landscape.counterfactual.validators import VALIDATOR_VERSION
    from legal_landscape.data.sanitize import SANITIZER_VERSION

    return {
        "prompt_version": PROMPT_VERSION,
        "sanitizer_version": SANITIZER_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "parent_text_sha256": text_sha256(parent_text),
        "target_defendant": defendant,
    }
