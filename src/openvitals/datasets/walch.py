"""Walch et al. 2019, "Motion and heart rate from a wrist-worn wearable and
labeled sleep from polysomnography" (PhysioNet sleep-accel 1.0.0).

31 subjects, one night each. Apple Watch acceleration (g) and PPG heart rate
(bpm), PSG stages in 30 s epochs. All times are seconds since PSG start.

Layout:
    motion/<id>_acceleration.txt      t x y z
    heart_rate/<id>_heartrate.txt     t bpm
    labels/<id>_labeled_sleep.txt     t stage   (0 wake, 1 N1, 2 N2, 3 N3, 4 N4, 5 REM, -1 unscored)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import stages
from ..schema import EPOCH_SECONDS, Night
from ._io import cache_path, read_numeric

NAME = "walch"
URL = "https://physionet.org/files/sleep-accel/1.0.0/"
MIRROR = "https://physionet-open.s3.amazonaws.com/sleep-accel/1.0.0/"  # AWS Open Data copy
LICENSE = "ODC-By 1.0"

# Native code -> canonical. N4 (old R&K scoring) folds into N3.
_STAGE_MAP = {0: stages.WAKE, 1: stages.N1, 2: stages.N2, 3: stages.N3, 4: stages.N3, 5: stages.REM}


def subject_ids(root: Path) -> list[str]:
    return sorted(p.name.split("_")[0] for p in (Path(root) / "labels").glob("*_labeled_sleep.txt"))


def load(root: Path, cache_dir: Path | None = None, subjects: list[str] | None = None) -> list[Night]:
    root = Path(root)
    if not (root / "labels").is_dir():
        raise FileNotFoundError(f"{root} does not look like sleep-accel (no labels/ folder)")
    return [_load_one(root, sid, cache_dir) for sid in (subjects or subject_ids(root))]


def _load_one(root: Path, sid: str, cache_dir: Path | None) -> Night:
    cached = cache_path(cache_dir, NAME, str(root.resolve()), sid) if cache_dir else None
    if cached and cached.exists():
        z = np.load(cached)
        motion, hr, labels = z["motion"], z["hr"], z["labels"]
    else:
        motion = read_numeric(root / "motion" / f"{sid}_acceleration.txt", 4)
        hr = read_numeric(root / "heart_rate" / f"{sid}_heartrate.txt", 2)
        labels = labels_to_epochs(read_numeric(root / "labels" / f"{sid}_labeled_sleep.txt", 2))
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cached, motion=motion, hr=hr, labels=labels)
    return Night(NAME, sid, "1", motion, hr, labels, meta={"time_ref": "psg_start"})


def labels_to_epochs(rows: np.ndarray) -> np.ndarray:
    """Turn (t, stage) rows into a dense per-epoch canonical array starting at t=0.

    Rows before t=0 are dropped; missing epochs become UNKNOWN.
    """
    t = rows[:, 0]
    code = rows[:, 1].astype(int)
    keep = t >= 0
    t, code = t[keep], code[keep]
    if t.size == 0:
        return np.empty(0, dtype=np.int8)
    idx = np.floor(t / EPOCH_SECONDS + 1e-6).astype(int)
    out = np.full(idx.max() + 1, stages.UNKNOWN, dtype=np.int8)
    for i, c in zip(idx, code):
        out[i] = _STAGE_MAP.get(c, stages.UNKNOWN)
    return out
