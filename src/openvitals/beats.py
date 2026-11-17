"""Beat detection from raw waveforms: ECG R-peaks and wrist PPG (BVP) pulse beats.

Both return beat times in seconds (sub-sample, by parabolic interpolation), so the
interval series can go straight into `recovery.clean_ibi` / `rmssd`.

PPG beats are the point of steepest upstroke (maximum first derivative), which is
more stable than the systolic peak under motion and changing pulse shape.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

MIN_RR_S = 0.30   # 200 bpm
WIN_S = 10.0      # adaptive threshold window


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    sos = signal.butter(3, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def _fill_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def _parabolic(y: np.ndarray, i: np.ndarray) -> np.ndarray:
    """Sub-sample offset of a local extremum at index i."""
    i = i[(i > 0) & (i < len(y) - 1)]
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    return i + np.clip(off, -0.5, 0.5)


def _adaptive_peaks(x: np.ndarray, fs: float, frac: float) -> np.ndarray:
    """Peaks of x at least MIN_RR_S apart and above `frac` of the local (10 s) 98th percentile."""
    n = len(x)
    w = max(1, int(WIN_S * fs))
    starts = np.arange(0, n, w)
    ref = np.array([np.percentile(x[s:s + w], 98) for s in starts])
    thr = frac * np.repeat(ref, w)[:n]
    pk, _ = signal.find_peaks(x, distance=max(1, int(MIN_RR_S * fs)))
    return pk[x[pk] > thr[pk]]


def ecg_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """R-peak times (s from sample 0). Polarity is detected automatically."""
    x = _bandpass(_fill_nan(ecg), fs, 5.0, min(30.0, 0.45 * fs))
    if np.percentile(x, 0.5) < -np.percentile(x, 99.5):  # inverted lead: R deflects down
        x = -x
    pk = _adaptive_peaks(np.clip(x, 0, None), fs, 0.35)
    return _parabolic(x, pk) / fs


def ppg_beats(bvp: np.ndarray, fs: float) -> np.ndarray:
    """Pulse beat times (s from sample 0): maximum upstroke slope of each pulse."""
    x = _bandpass(_fill_nan(bvp), fs, 0.5, 4.0)
    d = np.gradient(x)
    # 1. systolic peaks: keep only prominent ones, so the secondary (diastolic) wave
    #    isn't counted as a beat
    pk, props = signal.find_peaks(x, distance=max(1, int(MIN_RR_S * fs)), prominence=0)
    if pk.size == 0:
        return np.empty(0)
    prom = props["prominences"]
    ref = np.array([np.percentile(prom[max(0, i - 7):i + 8], 90) for i in range(len(prom))])
    pk = pk[prom > 0.4 * ref]
    # 2. beat time = steepest upstroke in the 350 ms before each systolic peak
    look = max(2, int(0.35 * fs))
    up = np.array([p - look + int(np.argmax(d[max(0, p - look):p + 1])) + (0 if p >= look else look - p)
                   for p in pk])
    return _parabolic(d, np.unique(up)) / fs
