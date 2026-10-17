"""Small readers shared by the loaders."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def read_numeric(path: Path, ncols: int, header: bool | None = None) -> np.ndarray:
    """Read a numeric text table, sniffing the delimiter (comma or whitespace).

    `header=None` auto-detects a non-numeric first line.
    """
    with open(path) as f:
        first = f.readline()
    sep = "," if "," in first else r"\s+"
    if header is None:
        header = not _is_numeric_row(first, sep)
    df = pd.read_csv(
        path,
        sep=sep,
        header=0 if header else None,
        usecols=range(ncols),
        dtype=np.float64,
        on_bad_lines="skip",
    )
    return df.to_numpy()


def _is_numeric_row(line: str, sep: str) -> bool:
    parts = [p for p in (line.strip().split(",") if sep == "," else line.split()) if p]
    try:
        [float(p) for p in parts]
    except ValueError:
        return False
    return bool(parts)


def cache_path(cache_dir: Path, *parts: str) -> Path:
    digest = hashlib.sha1("/".join(parts).encode()).hexdigest()[:16]
    return Path(cache_dir) / f"{parts[0]}-{digest}.npz"
