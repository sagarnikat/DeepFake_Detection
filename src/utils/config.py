"""YAML configuration loading with attribute-style access.

Usage
-----
>>> from src.utils.config import load_config
>>> cfg = load_config("configs/train_ffpp.yaml")
>>> cfg.train.epochs
20
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """A dict whose keys are also accessible as attributes (recursively)."""

    def __init__(self, mapping: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        if mapping:
            for key, value in mapping.items():
                self[key] = self._wrap(value)
        for key, value in kwargs.items():
            self[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, Mapping) and not isinstance(value, Config):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(v) for v in value]
        return value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = Config._wrap(value)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(dict(self))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    result: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def load_config(
    path: str | os.PathLike[str],
    overrides: Mapping[str, Any] | None = None,
    default_path: str | os.PathLike[str] | None = None,
) -> Config:
    """Load a YAML config.

    Parameters
    ----------
    path:
        YAML file to load.
    overrides:
        Optional mapping merged on top of the file contents.
    default_path:
        Optional base config merged *under* ``path`` (e.g. ``configs/base.yaml``).
    """
    merged: Dict[str, Any] = {}
    if default_path is not None:
        with open(_resolve(default_path), "r", encoding="utf-8") as fh:
            merged = yaml.safe_load(fh) or {}

    with open(_resolve(path), "r", encoding="utf-8") as fh:
        merged = deep_merge(merged, yaml.safe_load(fh) or {})

    if overrides:
        merged = deep_merge(merged, overrides)

    return Config(merged)


def dump_config(cfg: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    """Write a config back to YAML."""
    out = _resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False)
