"""Configuration loading with deterministic override precedence."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "LEGAL_LANDSCAPE_"


def _coerce(value: str) -> Any:
    """Parse environment and CLI strings using YAML scalar semantics."""
    parsed = yaml.safe_load(value)
    return parsed if not isinstance(parsed, dict | list) else value


def _set_path(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.lower().split(".")
    cursor = config
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot override nested key below scalar: {dotted_key}")
        cursor = child
    cursor[parts[-1]] = value


def load_config(
    path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML, then apply environment and explicit overrides.

    Environment keys use ``LEGAL_LANDSCAPE_SECTION__KEY`` syntax.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("configuration root must be a mapping")
    config: dict[str, Any] = deepcopy(loaded)
    for key, raw in (os.environ if env is None else env).items():
        if key.startswith(ENV_PREFIX):
            dotted = key[len(ENV_PREFIX) :].replace("__", ".")
            _set_path(config, dotted, _coerce(raw))
    for key, value in (overrides or {}).items():
        if value is not None:
            _set_path(config, key, value)
    return config


def parse_overrides(values: list[str]) -> dict[str, Any]:
    """Parse repeated ``KEY=VALUE`` command-line overrides."""
    result: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"override must be KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        result[key] = _coerce(value)
    return result
