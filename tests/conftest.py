"""Synthetic nights written in each dataset's on-disk format.

Signals depend on the stage (wake: movement + high HR; deep: still + low HR;
REM: still + jittery HR), so a working pipeline must beat the majority baseline.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

EPOCH = 30
# Walch native codes: 0 wake, 1 N1, 2 N2, 3 N3, 5 REM
WALCH_CODES = [0, 1, 2, 3, 5]
# BIDSleep native codes: 0 wake, 1 N1, 2 N2, 3 N3, 4 REM, 5 unknown
BIDS_CODES = [0, 1, 2, 3, 4]


def hypnogram(rng, n_epochs: int) -> np.ndarray:
    """Canonical-index hypnogram (0 wake, 1 N1, 2 N2, 3 N3, 4 REM) in runs."""
    out, stage = [], 0
    while len(out) < n_epochs:
        out += [stage] * int(rng.integers(6, 40))
        stage = int(rng.choice([0, 1, 2, 2, 3, 4], p=[0.15, 0.1, 0.3, 0.15, 0.15, 0.15]))
    return np.array(out[:n_epochs])


def signals(rng, hyp: np.ndarray, t0: float = 0.0, fs: float = 2.0, hr_dt: float = 5.0):
    n = len(hyp)
    t = np.arange(-60, n * EPOCH, 1 / fs)
    ep = np.clip(np.floor(t / EPOCH).astype(int), 0, n - 1)
    move = np.array([0.15, 0.03, 0.01, 0.003, 0.004])[hyp[ep]]
    x = rng.normal(0, move)
    y = rng.normal(0, move)
    z = 1.0 + rng.normal(0, move)
    motion = np.column_stack([t + t0, x, y, z])
    th = np.arange(-60, n * EPOCH, hr_dt)
    eh = np.clip(np.floor(th / EPOCH).astype(int), 0, n - 1)
    base = np.array([72, 64, 60, 55, 63])[hyp[eh]]
    jitter = np.array([4, 2, 1.5, 1, 5])[hyp[eh]]
    hr = np.column_stack([th + t0, base + rng.normal(0, jitter)])
    return motion, hr


@pytest.fixture(scope="session")
def walch_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("sleep-accel")
    for d in ("motion", "heart_rate", "labels"):
        (root / d).mkdir()
    rng = np.random.default_rng(1)
    for s in range(8):
        sid = str(1000 + s)
        hyp = hypnogram(rng, 600)
        motion, hr = signals(rng, hyp)
        np.savetxt(root / "motion" / f"{sid}_acceleration.txt", motion, fmt="%.6f", delimiter=" ")
        np.savetxt(root / "heart_rate" / f"{sid}_heartrate.txt", hr, fmt="%.3f", delimiter=",")
        codes = np.array(WALCH_CODES)[hyp]
        codes[:3] = -1  # unscored lead-in, as in the real files
        labels = np.column_stack([np.arange(len(hyp)) * EPOCH, codes])
        np.savetxt(root / "labels" / f"{sid}_labeled_sleep.txt", labels, fmt="%d", delimiter=" ")
    return root


@pytest.fixture(scope="session")
def bidsleep_root(tmp_path_factory):
    import scipy.io

    root = tmp_path_factory.mktemp("bidsleep")
    rng = np.random.default_rng(2)
    eastern = ZoneInfo("America/New_York")
    for s in range(6):
        for night in (1, 2, 3):
            d = root / f"Bidslab{s:02d}" / str(night)
            d.mkdir(parents=True)
            start = datetime(2024, 3, 9 + night, 23, 15, 0, tzinfo=eastern)  # spans the DST change
            t0 = start.timestamp()
            hyp = hypnogram(rng, 500)
            motion, hr = signals(rng, hyp, t0=t0, fs=1.0)
            with open(d / "motion.csv", "w") as f:
                f.write("Timestamp,x,y,z\n")
                np.savetxt(f, motion, fmt="%.6f", delimiter=",")
            np.savetxt(d / "hr.csv", hr, fmt="%.4f", delimiter=",")
            expert = np.array(BIDS_CODES)[hyp].astype(float)
            expert[-2:] = 5  # unknown
            scipy.io.savemat(d / "labels.mat", {
                "recStart": start.strftime("%Y-%m-%d %H:%M:%S"),
                "dreem_label": expert.copy(),
                "expert_label": expert,
            })
    return root
