"""The device-agnostic record every loader produces and every model consumes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPOCH_SECONDS = 30


@dataclass
class Night:
    """One recorded night, aligned to 30-second scoring epochs.

    Time is in seconds relative to the start of epoch 0, so epoch k covers
    [30k, 30k + 30). Signals may start before 0 or end after the last epoch.

    motion: (n, 4) array of t, x, y, z (acceleration in g)
    hr:     (m, 2) array of t, bpm
    labels: (n_epochs,) canonical stage codes (see `stages`), -1 = unscored
    """

    dataset: str
    subject_id: str
    night_id: str
    motion: np.ndarray
    hr: np.ndarray
    labels: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.subject_id}/{self.night_id}"

    @property
    def n_epochs(self) -> int:
        return int(self.labels.shape[0])

    def __post_init__(self) -> None:
        self.motion = _as_2d(self.motion, 4, "motion")
        self.hr = _as_2d(self.hr, 2, "hr")
        self.labels = np.asarray(self.labels, dtype=np.int8)
        # Sort by time; some exports are not monotonic.
        if len(self.motion):
            self.motion = self.motion[np.argsort(self.motion[:, 0], kind="stable")]
        if len(self.hr):
            self.hr = self.hr[np.argsort(self.hr[:, 0], kind="stable")]


def _as_2d(a: np.ndarray, ncols: int, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return np.empty((0, ncols))
    if a.ndim != 2 or a.shape[1] != ncols:
        raise ValueError(f"{name} must have shape (n, {ncols}), got {a.shape}")
    return a[np.all(np.isfinite(a), axis=1)]
