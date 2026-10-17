"""Dataset registry. Each module exposes NAME, URL, LICENSE, subject_ids(root)
and load(root, cache_dir, subjects=None)."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

from ..schema import Night
from . import bidsleep, walch

REGISTRY = {m.NAME: m for m in (walch, bidsleep)}


def load(name: str, root: Path, cache_dir: Path | None = None, compact: bool = False, **kw) -> list[Night]:
    """compact=True returns nights with every feature set computed and the raw
    signals dropped. With a cache_dir, compacted nights are cached on disk per
    subject, so later runs never read raw signals (Walch raw motion is ~1.7 GB
    in memory, and macOS keeps freed memory resident)."""
    if name not in REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(REGISTRY)}")
    mod, root = REGISTRY[name], Path(root)
    if not compact:
        return mod.load(root, cache_dir=cache_dir, **kw)
    from .. import features

    subjects = kw.pop("subjects", None) or mod.subject_ids(root)
    all_files = sorted(p for p in root.rglob("*") if p.is_file())
    nights = []
    for sid in subjects:
        path = _feature_cache_path(cache_dir, name, root, sid, kw, all_files, features.FEATURE_VERSION)
        if path and path.exists():
            with open(path, "rb") as f:
                nights += pickle.load(f)
            continue
        loaded = [features.compact(n) for n in mod.load(root, cache_dir=cache_dir, subjects=[sid], **kw)]
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(loaded, f)
        nights += loaded
    return nights


def _feature_cache_path(cache_dir, name, root, sid, kw, all_files, version) -> Path | None:
    if cache_dir is None:
        return None
    mine = [p for p in all_files if sid in p.relative_to(root).parts or p.name.startswith(sid + "_")]
    sig = "|".join(f"{p.relative_to(root)}:{p.stat().st_size}" for p in mine)
    key = f"{root.resolve()}|{sid}|{sorted(kw.items())}|{version}|{sig}"
    return Path(cache_dir) / f"features-{name}-{hashlib.sha1(key.encode()).hexdigest()[:16]}.pkl"
