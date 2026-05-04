"""Deep-merge nested dicts for YAML ablations."""

from __future__ import annotations

from typing import Any


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, val in overrides.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = deep_merge(out[key], val)  # type: ignore[arg-type]
        else:
            out[key] = val
    return out
