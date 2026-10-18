"""BIDSleep Apple Watch dataset (PhysioNet bidsleep-dataset 1.0.1).

47 subjects, 3-7 free-living nights each. Apple Watch acceleration (g) and
HealthKit heart rate (~0.2 Hz); labels from a Dreem 2 EEG headband, with
expert-corrected stages. Sensor timestamps are Unix seconds; label epoch k
(1-based) covers [recStart + 30(k-1), recStart + 30k), recStart in US Eastern.

Layout:
    Bidslab<XX>/<night>/motion.csv   header: Timestamp,x,y,z
    Bidslab<XX>/<night>/hr.csv       no header: unix_t, bpm
    Bidslab<XX>/<night>/labels.mat   recStart, dreem_label, expert_label
                                     (0 wake, 1 N1, 2 N2, 3 N3, 4 REM, 5 unknown)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .. import stages
from ..schema import Night
from ._io import cache_path, read_numeric

NAME = "bidsleep"
URL = "https://physionet.org/files/bidsleep-dataset/1.0.1/"
LICENSE = "see PhysioNet page (Open Access)"
EASTERN = ZoneInfo("America/New_York")

_STAGE_MAP = {0: stages.WAKE, 1: stages.N1, 2: stages.N2, 3: stages.N3, 4: stages.REM}


def night_dirs(root: Path) -> list[Path]:
    out = [p.parent for p in Path(root).glob("Bidslab*/*/labels.mat")]
    return sorted(out, key=lambda p: (p.parent.name, _int_or_str(p.name)))


def manifest_subject(rel: str) -> str | None:
    head = rel.split("/")[0]
    return head if head.startswith("Bidslab") and "/" in rel else None


def manifest_night(rel: str) -> str | None:
    """Night folder a dataset file belongs to (Bidslab00/<night>/file)."""
    parts = rel.split("/")
    return parts[1] if len(parts) >= 3 else None


def subject_ids(root: Path) -> list[str]:
    return sorted({d.parent.name for d in night_dirs(root)})


def load(
    root: Path,
    cache_dir: Path | None = None,
    subjects: list[str] | None = None,
    label_source: str = "expert",
) -> list[Night]:
    """label_source: "expert" (falls back to dreem if missing) or "dreem"."""
    dirs = night_dirs(root)
    if not dirs:
        raise FileNotFoundError(f"{root} does not look like BIDSleep (no Bidslab*/<night>/labels.mat)")
    if subjects:
        dirs = [d for d in dirs if d.parent.name in set(subjects)]
    return [_load_one(d, cache_dir, label_source) for d in dirs]


def _load_one(d: Path, cache_dir: Path | None, label_source: str) -> Night:
    sid, nid = d.parent.name, d.name
    has_motion = (d / "motion.csv").exists()
    cached = (cache_path(cache_dir, NAME, str(d.resolve()), label_source, f"motion={has_motion}")
              if cache_dir else None)
    if cached and cached.exists():
        z = np.load(cached)
        motion, hr, labels, start = z["motion"], z["hr"], z["labels"], float(z["start"])
    else:
        rec_start, raw_labels = read_labels(d / "labels.mat", label_source)
        start = rec_start.timestamp()
        labels = np.array([_STAGE_MAP.get(int(c), stages.UNKNOWN) for c in raw_labels], dtype=np.int8)
        # motion.csv is ~95% of the 27.9 GB download; allow HR-only copies.
        motion = read_numeric(d / "motion.csv", 4) if has_motion else np.empty((0, 4))
        hr = read_numeric(d / "hr.csv", 2, header=False)
        # Re-reference Unix time to the start of epoch 0.
        if len(motion):
            motion[:, 0] -= start
        hr[:, 0] -= start
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cached, motion=motion, hr=hr, labels=labels, start=start)
    return Night(NAME, sid, nid, motion, hr, labels,
                 meta={"time_ref": "recStart", "unix_start": start, "label_source": label_source})


def read_labels(path: Path, label_source: str = "expert") -> tuple[datetime, np.ndarray]:
    mat = _loadmat(path)
    keys = ["expert_label", "dreem_label"] if label_source == "expert" else ["dreem_label"]
    for k in keys:
        if k in mat and np.size(mat[k]):
            labels = np.asarray(mat[k]).astype(float).ravel()
            break
    else:
        raise KeyError(f"{path}: none of {keys} present")
    labels = np.where(np.isfinite(labels), labels, 5).astype(int)
    return parse_rec_start(mat["recStart"]), labels


def parse_rec_start(value) -> datetime:
    """recStart is documented as human-readable US Eastern time. Also accept a
    MATLAB datenum, in case some files store it numerically."""
    v = np.asarray(value)
    if v.dtype.kind in "fiu":
        datenum = float(v.ravel()[0])
        naive = datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)
    else:
        text = "".join(np.atleast_1d(v).astype(str).ravel()).strip()
        naive = pd.to_datetime(text).to_pydatetime()
    if naive.tzinfo is None:
        return naive.replace(tzinfo=EASTERN)
    return naive


def _loadmat(path: Path) -> dict:
    import scipy.io

    try:
        return scipy.io.loadmat(path, squeeze_me=True)
    except NotImplementedError:  # MATLAB v7.3 = HDF5
        try:
            import h5py
        except ImportError as e:
            raise ImportError(f"{path} is MATLAB v7.3; install openvitals[mat73]") from e
        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                arr = np.asarray(f[k])
                if arr.dtype.kind in "u" and k == "recStart":  # char data stored as uint16
                    arr = np.array("".join(map(chr, arr.ravel())))
                out[k] = arr
        return out


def _int_or_str(s: str):
    return (0, int(s)) if s.isdigit() else (1, s)
