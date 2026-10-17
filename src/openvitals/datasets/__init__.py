"""Dataset registry. Each module exposes NAME, URL, LICENSE and load(root, cache_dir)."""

from __future__ import annotations

from pathlib import Path

from ..schema import Night
from . import bidsleep, walch

REGISTRY = {m.NAME: m for m in (walch, bidsleep)}


def load(name: str, root: Path, cache_dir: Path | None = None, **kw) -> list[Night]:
    if name not in REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name].load(Path(root), cache_dir=cache_dir, **kw)
