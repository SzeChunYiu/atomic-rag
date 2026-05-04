"""Strict YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def load_yaml(path: Path, model: type[T]) -> T:
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        msg = f"{path}: root must be a mapping"
        raise ValueError(msg)
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        msg = f"{path}: config schema mismatch:\n{exc}"
        raise ValueError(msg) from exc
