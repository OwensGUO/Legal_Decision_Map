"""Canonical labels for Criminal Law articles and paragraphs."""

from __future__ import annotations

import re
from collections.abc import Iterable


def normalize_criminal_article(value: object) -> str:
    """Normalize source labels such as ``234`` or ``303-2``.

    The source datasets omit the statute name and use hyphens for nested
    paragraphs. This project currently supervises Criminal Law provisions only,
    so the statute namespace is made explicit in the normalized label.
    """
    raw = str(value).strip()
    if not raw:
        raise ValueError("article label must not be empty")
    parts = re.split(r"[-:./]", raw)
    if any(not part.isdigit() or int(part) <= 0 for part in parts):
        raise ValueError(f"invalid Criminal Law article label: {value!r}")
    return "criminal_law:" + ":".join(str(int(part)) for part in parts)


def normalize_criminal_articles(values: Iterable[object]) -> tuple[str, ...]:
    """Normalize and de-duplicate article labels while preserving source order."""
    return tuple(dict.fromkeys(normalize_criminal_article(value) for value in values))
