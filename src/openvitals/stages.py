"""Canonical sleep stage codes and the label schemes we evaluate on.

Every loader maps its native codes onto CANONICAL. Models and metrics then work
in a *scheme* (2-, 3- or 4-class), produced by `to_scheme`.
"""

from __future__ import annotations

import numpy as np

UNKNOWN = -1
WAKE, N1, N2, N3, REM = 0, 1, 2, 3, 4

CANONICAL_NAMES = {WAKE: "wake", N1: "N1", N2: "N2", N3: "N3", REM: "REM"}

# scheme name -> (class names, mapping from canonical code to scheme index)
SCHEMES: dict[str, tuple[list[str], dict[int, int]]] = {
    "2class": (["wake", "sleep"], {WAKE: 0, N1: 1, N2: 1, N3: 1, REM: 1}),
    "3class": (["wake", "nrem", "rem"], {WAKE: 0, N1: 1, N2: 1, N3: 1, REM: 2}),
    "4class": (["wake", "light", "deep", "rem"], {WAKE: 0, N1: 1, N2: 1, N3: 2, REM: 3}),
}


def class_names(scheme: str) -> list[str]:
    return SCHEMES[_check(scheme)][0]


def to_scheme(labels: np.ndarray, scheme: str) -> np.ndarray:
    """Map canonical labels to scheme indices. Unknown stays -1."""
    mapping = SCHEMES[_check(scheme)][1]
    out = np.full(labels.shape, UNKNOWN, dtype=np.int8)
    for src, dst in mapping.items():
        out[labels == src] = dst
    return out


def sleep_index(scheme: str) -> set[int]:
    """Scheme indices that count as 'asleep' (everything except wake)."""
    return set(range(1, len(class_names(scheme))))


def _check(scheme: str) -> str:
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {sorted(SCHEMES)}")
    return scheme
