"""Per-epoch features from raw wrist motion and heart rate.

Deliberately simple and portable (numpy only), so the same features can be
reimplemented on-device. Every feature is computed from one night alone; no
information crosses nights or subjects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import EPOCH_SECONDS, Night

MOTION_FEATURES = [
    "enmo_log", "mag_std_log", "zangle_change",
    "enmo_log_w5", "enmo_log_w11", "enmo_log_w21", "zangle_change_w11",
]
HR_FEATURES = ["hr_z", "hr_z_w5", "hr_std_w5", "hr_std_w11", "hr_diff"]
TIME_FEATURES = ["hours_since_start"]
INPUTS = ("motion", "hr", "motion+hr")
# Bump whenever feature definitions change; it invalidates on-disk feature caches.
FEATURE_VERSION = 1


@dataclass
class EpochFeatures:
    X: np.ndarray            # (n_epochs, n_features)
    names: list[str]
    motion_cov: np.ndarray   # bool, epoch has any motion samples
    hr_cov: np.ndarray       # bool, epoch has any HR samples

    @property
    def scorable(self) -> np.ndarray:
        """Epochs with at least some sensor data."""
        return self.motion_cov | self.hr_cov


def extract(night: Night, inputs: str = "motion+hr") -> EpochFeatures:
    """Features for one night; memoised on the Night since folds reuse nights."""
    if inputs not in INPUTS:
        raise ValueError(f"inputs must be one of {INPUTS}")
    cache = night.__dict__.setdefault("_feature_cache", {})
    if inputs not in cache:
        if night.meta.get("compacted"):
            raise RuntimeError(f"{night.key} was compacted; raw signals are gone")
        cache[inputs] = _extract(night, inputs)
    return cache[inputs]


def compact(night: Night) -> Night:
    """Compute every feature set, then release the raw signals. Models only
    read features, so a compacted night behaves the same everywhere."""
    for inputs in INPUTS:
        extract(night, inputs)
    night.motion = np.empty((0, 4))
    night.hr = np.empty((0, 2))
    night.meta["compacted"] = True
    return night


def _extract(night: Night, inputs: str) -> EpochFeatures:
    n = night.n_epochs
    cols: dict[str, np.ndarray] = {}

    enmo, mag_std, zangle, motion_cov = _motion_per_epoch(night.motion, n)
    enmo_log = np.log1p(1000 * enmo)
    zchange = np.abs(np.diff(zangle, prepend=zangle[:1]))
    cols["enmo_log"] = enmo_log
    cols["mag_std_log"] = np.log1p(1000 * mag_std)
    cols["zangle_change"] = zchange
    for w in (5, 11, 21):
        cols[f"enmo_log_w{w}"] = _centered_mean(enmo_log, w)
    cols["zangle_change_w11"] = _centered_mean(zchange, 11)

    hr_mean, hr_cov = _per_epoch_mean(night.hr[:, 0], night.hr[:, 1], n)
    if "hr" in inputs:
        hr = _interp_missing(hr_mean)
        med = np.nanmedian(hr) if np.isfinite(hr).any() else 0.0
        scale = np.nanstd(hr) if np.isfinite(hr).any() else 1.0
        hr_z = (hr - med) / (scale if scale > 1e-6 else 1.0)
        hr_z = np.nan_to_num(hr_z)
        cols["hr_z"] = hr_z
        cols["hr_z_w5"] = _centered_mean(hr_z, 5)
        cols["hr_std_w5"] = _centered_std(hr_z, 5)
        cols["hr_std_w11"] = _centered_std(hr_z, 11)
        cols["hr_diff"] = np.abs(np.diff(hr_z, prepend=hr_z[:1]))

    cols["hours_since_start"] = np.arange(n) * EPOCH_SECONDS / 3600.0

    names = ((MOTION_FEATURES if "motion" in inputs else [])
             + (HR_FEATURES if "hr" in inputs else []) + TIME_FEATURES)
    X = np.column_stack([cols[c] for c in names]) if n else np.empty((0, len(names)))
    return EpochFeatures(X, names, motion_cov, hr_cov)


def _epoch_index(t: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.floor(t / EPOCH_SECONDS).astype(np.int64)
    keep = (idx >= 0) & (idx < n)
    return idx[keep], keep


def _per_epoch_mean(t: np.ndarray, v: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    idx, keep = _epoch_index(t, n)
    s = np.bincount(idx, weights=v[keep], minlength=n)
    c = np.bincount(idx, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return mean, c > 0


def _motion_per_epoch(motion: np.ndarray, n: int):
    if len(motion) == 0:
        z = np.zeros(n)
        return z, z.copy(), z.copy(), np.zeros(n, dtype=bool)
    t, x, y, zz = motion.T
    mag = np.sqrt(x * x + y * y + zz * zz)
    enmo = np.maximum(mag - 1.0, 0.0)
    zangle = np.degrees(np.arctan2(zz, np.sqrt(x * x + y * y)))
    idx, keep = _epoch_index(t, n)
    c = np.bincount(idx, minlength=n)
    cov = c > 0
    safe = np.maximum(c, 1)
    enmo_m = np.bincount(idx, weights=enmo[keep], minlength=n) / safe
    m1 = np.bincount(idx, weights=mag[keep], minlength=n) / safe
    m2 = np.bincount(idx, weights=mag[keep] ** 2, minlength=n) / safe
    mag_std = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    za = np.bincount(idx, weights=zangle[keep], minlength=n) / safe
    za = _interp_missing(np.where(cov, za, np.nan))
    return enmo_m, mag_std, np.nan_to_num(za), cov


def _interp_missing(a: np.ndarray) -> np.ndarray:
    a = a.astype(float).copy()
    ok = np.isfinite(a)
    if ok.sum() == 0:
        return a
    if ok.sum() < len(a):
        a[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), a[ok])
    return a


def _centered_mean(a: np.ndarray, w: int) -> np.ndarray:
    if len(a) == 0:
        return a
    k = np.ones(w) / w
    pad = w // 2
    return np.convolve(np.pad(a, pad, mode="edge"), k, mode="valid")


def _centered_std(a: np.ndarray, w: int) -> np.ndarray:
    if len(a) == 0:
        return a
    m = _centered_mean(a, w)
    m2 = _centered_mean(a * a, w)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))
