"""DREAMT: Dataset for Real-time sleep stage EstimAtion using Multisensor wearable
Technology (PhysioNet dreamt 2.2.0, restricted access).

100 adults, one lab night each (many with sleep apnea). Empatica E4 wristband
(BVP, accelerometer, temperature, EDA, HR, IBI) recorded alongside PSG; stages
scored by sleep technicians in 30 s epochs.

Layout (data_64Hz/, everything resampled to 64 Hz and forward-filled):
    S<NNN>_whole_df.csv   TIMESTAMP (s from E4 start), BVP, ACC_X, ACC_Y, ACC_Z (1/64 g),
                          TEMP, EDA, HR (bpm, 1 Hz from the E4), IBI (s), Sleep_Stage,
                          apnea event columns
    Sleep_Stage: P (preparation, before lights out), W, N1, N2, N3, R, Missing

Epoch 0 starts at the first non-P row, so each night is aligned to the PSG
scoring. Accelerometer is kept at 32 Hz (the E4's native rate).

Licence: PhysioNet Restricted Health Data Use Agreement 1.5.0. Research use
only: OpenVitals uses DREAMT to *test* models, never to train the shipped one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import stages
from ..schema import EPOCH_SECONDS, Night
from ._io import cache_path

NAME = "dreamt"
URL = "https://physionet.org/files/dreamt/2.2.0/"
LICENSE = "PhysioNet Restricted Health Data Use Agreement 1.5.0 (research use only)"
SUBDIR = "data_64Hz"
ACC_UNITS_PER_G = 64.0

_STAGE_MAP = {"W": stages.WAKE, "N1": stages.N1, "N2": stages.N2, "N3": stages.N3, "R": stages.REM}
_COLS = ["TIMESTAMP", "ACC_X", "ACC_Y", "ACC_Z", "HR", "IBI", "Sleep_Stage"]


def _files(root: Path) -> dict[str, Path]:
    root = Path(root)
    folder = root / SUBDIR if (root / SUBDIR).is_dir() else root
    return {p.name.split("_")[0]: p for p in sorted(folder.glob("S*_whole_df.csv"))}


def manifest_subject(rel: str) -> str | None:
    folder, _, fname = rel.partition("/")
    return fname.split("_")[0] if folder == SUBDIR and fname.endswith("_whole_df.csv") else None


def subject_ids(root: Path) -> list[str]:
    return list(_files(root))


def load(root: Path, cache_dir: Path | None = None, subjects: list[str] | None = None) -> list[Night]:
    files = _files(root)
    if not files:
        raise FileNotFoundError(f"{root} has no DREAMT S*_whole_df.csv files (looked in {SUBDIR}/ too)")
    return [_load_one(files[sid], sid, cache_dir) for sid in (subjects or list(files))]


def _load_one(path: Path, sid: str, cache_dir: Path | None) -> Night:
    cached = cache_path(cache_dir, NAME, str(path.resolve()), str(path.stat().st_size)) if cache_dir else None
    if cached and cached.exists():
        z = np.load(cached)
        motion, hr, labels, beats, t0 = z["motion"], z["hr"], z["labels"], z["beats"], float(z["t0"])
    else:
        motion, hr, labels, beats, t0 = parse(read_table(path))
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cached, motion=motion, hr=hr, labels=labels, beats=beats, t0=t0)
    return Night(NAME, sid, "1", motion, hr, labels,
                 meta={"time_ref": "first_scored_epoch", "e4_offset_s": t0, "beats": beats})


def read_table(path: Path) -> pd.DataFrame:
    dtypes = {c: np.float64 for c in _COLS[:-1]}
    return pd.read_csv(path, usecols=_COLS, dtype={**dtypes, "Sleep_Stage": "category"})


def parse(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Raw table -> (motion, hr, labels, beats, t0), times relative to epoch 0.

    labels: each 30 s epoch takes the most common stage of its rows (Missing -> UNKNOWN).
    hr:     one sample per second (the E4 updates HR once a second).
    beats:  (k, 2) E4 beat times and intervals in s, from changes in the forward-filled
            IBI column. Consecutive identical intervals merge, so this is approximate.
    """
    t = df["TIMESTAMP"].to_numpy(np.float64)
    stage = df["Sleep_Stage"].astype(str).to_numpy()
    scored = np.flatnonzero(stage != "P")
    if scored.size == 0:
        raise ValueError("no scored rows (Sleep_Stage is all P)")
    t0 = float(t[scored[0]])
    rel = t - t0

    # labels: majority vote per epoch over scored rows
    code = np.full(t.shape, stages.UNKNOWN, dtype=np.int8)
    for name, c in _STAGE_MAP.items():
        code[stage == name] = c
    sel = scored
    ep = np.floor(rel[sel] / EPOCH_SECONDS + 1e-9).astype(np.int64)
    n_ep = int(ep.max()) + 1
    counts = np.zeros((n_ep, 6), dtype=np.int64)  # column 5 = unknown
    np.add.at(counts, (ep, np.where(code[sel] >= 0, code[sel], 5)), 1)
    labels = np.where(counts[:, :5].sum(1) * 2 > counts.sum(1), counts[:, :5].argmax(1), stages.UNKNOWN)
    labels = labels.astype(np.int8)

    # motion at 32 Hz, in g
    acc = df[["ACC_X", "ACC_Y", "ACC_Z"]].to_numpy(np.float64)[::2] / ACC_UNITS_PER_G
    motion = np.column_stack([rel[::2], acc])

    # HR once a second
    on_sec = np.abs(t - np.round(t)) < 1e-6
    hr = np.column_stack([rel[on_sec], df["HR"].to_numpy(np.float64)[on_sec]])
    hr = hr[np.isfinite(hr[:, 1]) & (hr[:, 1] > 0)]

    ibi = df["IBI"].to_numpy(np.float64)
    change = np.flatnonzero(np.isfinite(ibi) & (np.r_[np.nan, ibi[:-1]] != ibi))
    beats = np.column_stack([rel[change], ibi[change]]) if change.size else np.empty((0, 2))
    return motion, hr, labels, beats, t0
