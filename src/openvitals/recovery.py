"""Overnight resting heart rate and heart-rate variability (HRV).

Resting heart rate needs only the heart-rate samples `openvitals stage` already
uses, plus which epochs were asleep. HRV needs beat-to-beat intervals (IBIs):
heart rate sampled every few seconds averages the beat-to-beat variation away.

    resting_hr(hr_t, hr_bpm, asleep, unix_start)    lowest 5-min mean HR asleep, mean sleeping HR
    clean_ibi(ibi_ms)                               artifact mask (range + local-median deviation)
    rmssd(ibi_ms, ok)                               over successive pairs of clean beats only
    hrv_windows(beat_t, ibi_ms)                     RMSSD / SDNN / mean HR per 5-min window
    nightly_hrv(beat_t, ibi_ms, in_sleep)           median of clean 5-min windows while asleep

Methods follow common practice (e.g. Task Force of the ESC/NASPE 1996 for RMSSD and
SDNN; a local-median rule for artifacts). Validation is in scripts/validate_hrv.py.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .schema import EPOCH_SECONDS

IBI_MIN_MS, IBI_MAX_MS = 300.0, 2000.0  # 30-200 bpm
LOCAL_WINDOW = 11                       # beats in the rolling median
LOCAL_TOLERANCE = 0.20                  # flag beats >20% off the local median
HRV_WINDOW_S = 300                      # 5-minute windows, the usual short-term HRV unit
MIN_CLEAN = 0.80                        # windows need >=80% clean beats
RHR_WINDOW_EPOCHS = 10                  # 5 minutes of 30 s epochs
RHR_MIN_VALID = 8                       # of which at least 8 asleep
RHR_MIN_SAMPLES = 3                     # and at least 3 heart-rate samples in the window


# -- resting heart rate -----------------------------------------------------------

def resting_hr(hr_t: np.ndarray, hr_bpm: np.ndarray, asleep: np.ndarray, unix_start: float) -> dict:
    """Lowest 5-minute mean heart rate while asleep, and the mean over all sleep epochs.

    hr_t: absolute Unix seconds; asleep: bool per 30 s epoch from unix_start. A window
    counts if at least 8 of its 10 epochs are asleep and it holds a few HR samples, so
    devices that log once a minute (half the epochs empty) still get a value."""
    asleep = np.asarray(asleep, bool)
    n = len(asleep)
    idx = np.floor((np.asarray(hr_t, float) - unix_start) / EPOCH_SECONDS).astype(int)
    keep = (idx >= 0) & (idx < n)
    s = np.bincount(idx[keep], weights=np.asarray(hr_bpm, float)[keep], minlength=n) * asleep
    c = np.bincount(idx[keep], minlength=n) * asleep
    out = {"avg_sleep_hr": float(s.sum() / c.sum()) if c.sum() else None,
           "lowest_hr_5min": None, "lowest_hr_at": None, "sleep_hr_samples": int(c.sum())}
    k = RHR_WINDOW_EPOCHS
    if n < k:
        return out
    ones = np.ones(k)
    win_asleep = np.convolve(asleep.astype(float), ones, mode="valid")
    win_sum, win_n = np.convolve(s, ones, mode="valid"), np.convolve(c, ones, mode="valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where((win_asleep >= RHR_MIN_VALID) & (win_n >= RHR_MIN_SAMPLES), win_sum / win_n, np.nan)
    if np.isfinite(means).any():
        i = int(np.nanargmin(means))
        out["lowest_hr_5min"] = float(means[i])
        out["lowest_hr_at"] = float(unix_start + (i + k / 2) * EPOCH_SECONDS)
    return out


# -- heart-rate variability -------------------------------------------------------

def clean_ibi(ibi_ms: np.ndarray) -> np.ndarray:
    """True for plausible beats: inside 300-2000 ms and within 20% of the rolling median
    of the surrounding 11 intervals. Catches missed beats (one long interval), extra
    beats (two short), and ectopic short-long pairs."""
    ibi = np.asarray(ibi_ms, float)
    ok = np.isfinite(ibi) & (ibi >= IBI_MIN_MS) & (ibi <= IBI_MAX_MS)
    if ok.sum() < 3:
        return ok
    med = _rolling_median(np.where(ok, ibi, np.nan), LOCAL_WINDOW)
    with np.errstate(invalid="ignore"):
        ok &= np.abs(ibi - med) <= LOCAL_TOLERANCE * med
    return ok


def rmssd(ibi_ms: np.ndarray, ok: np.ndarray | None = None) -> float:
    """Root mean square of successive differences, over pairs where both beats are clean.
    A removed beat breaks the pair, so it never creates a spurious large difference."""
    ibi = np.asarray(ibi_ms, float)
    ok = np.ones(len(ibi), bool) if ok is None else np.asarray(ok, bool)
    pair = ok[1:] & ok[:-1]
    if pair.sum() < 2:
        return float("nan")
    d = np.diff(ibi)[pair]
    return float(np.sqrt(np.mean(d * d)))


def sdnn(ibi_ms: np.ndarray, ok: np.ndarray | None = None) -> float:
    ibi = np.asarray(ibi_ms, float)
    ok = np.ones(len(ibi), bool) if ok is None else np.asarray(ok, bool)
    return float(np.std(ibi[ok], ddof=1)) if ok.sum() > 2 else float("nan")


def hrv_windows(beat_t: np.ndarray, ibi_ms: np.ndarray, window_s: float = HRV_WINDOW_S,
                clean: bool = True) -> list[dict]:
    """Per non-overlapping window: start time, RMSSD, SDNN, mean HR, clean fraction.
    beat_t: time of each beat (s, any origin); ibi_ms: interval ending at that beat."""
    t = np.asarray(beat_t, float)
    ibi = np.asarray(ibi_ms, float)
    ok = clean_ibi(ibi) if clean else np.isfinite(ibi)
    if len(t) == 0:
        return []
    edges = np.arange(t[0], t[-1] + window_s, window_s)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b)
        if m.sum() < 30:  # too few beats for a 5-minute window
            continue
        w_ibi, w_ok = ibi[m], ok[m]
        out.append({"start": float(a), "n_beats": int(m.sum()), "clean_fraction": float(w_ok.mean()),
                    "rmssd_ms": rmssd(w_ibi, w_ok), "sdnn_ms": sdnn(w_ibi, w_ok),
                    "mean_hr": float(60000.0 / np.mean(w_ibi[w_ok])) if w_ok.any() else float("nan")})
    return out


def nightly_hrv(beat_t: np.ndarray, ibi_ms: np.ndarray, in_sleep: Callable[[float], bool] | None = None,
                window_s: float = HRV_WINDOW_S) -> dict:
    """Nightly HRV: median over clean 5-minute windows (>=80% clean beats) that fall
    asleep, as judged by `in_sleep(window_start)` (all windows if None)."""
    wins = hrv_windows(beat_t, ibi_ms, window_s)
    good = [w for w in wins if w["clean_fraction"] >= MIN_CLEAN and np.isfinite(w["rmssd_ms"])
            and (in_sleep is None or in_sleep(w["start"]))]
    out = {"n_windows": len(good), "n_windows_total": len(wins), "rmssd_ms": None, "ln_rmssd": None,
           "sdnn_ms": None, "clean_fraction": float(np.mean([w["clean_fraction"] for w in wins])) if wins else None}
    if good:
        r = float(np.median([w["rmssd_ms"] for w in good]))
        out.update(rmssd_ms=r, ln_rmssd=float(np.log(r)), sdnn_ms=float(np.median([w["sdnn_ms"] for w in good])))
    return out


def ibi_from_beats(beat_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Beat times (s) -> (time of each beat after the first, interval ending there in ms)."""
    t = np.asarray(beat_t, float)
    return t[1:], np.diff(t) * 1000.0


def _rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    """Centred rolling median ignoring NaNs (edges use the available neighbours)."""
    import warnings

    h = w // 2
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(x, h, constant_values=np.nan), w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows -> NaN
        return np.nanmedian(windows, axis=1)


# -- one night, from a staging result -------------------------------------------------

def night_recovery(result, hr_t: np.ndarray, hr_bpm: np.ndarray, beats: tuple | None = None,
                   apple_sessions: list[dict] | None = None) -> dict:
    """Resting HR and HRV for one staged night.

    result: an infer.StagingResult (stage per epoch, unix_start). beats: (beat_t, ibi_ms)
    from a beat-interval recording (chest strap); apple_sessions: from
    adapters.read_apple_health_hrv. HRV only uses beats/readings that fall asleep."""
    stage, start = np.asarray(result.stage), float(result.unix_start)
    asleep = stage > 0

    def in_sleep(ts: float) -> bool:
        i = int((ts - start) // EPOCH_SECONDS)
        return 0 <= i < len(stage) and bool(asleep[i])

    out = {"resting_hr": resting_hr(hr_t, hr_bpm, asleep, start), "hrv": None}
    end = start + EPOCH_SECONDS * len(stage)
    if beats is not None and len(beats[0]):
        t, ibi = np.asarray(beats[0], float), np.asarray(beats[1], float)
        m = (t >= start) & (t < end)
        if m.sum() >= 30:
            out["hrv"] = {"source": "beat intervals", **nightly_hrv(t[m], ibi[m], in_sleep)}
    elif apple_sessions:
        vals, sdnns = [], []
        for s in apple_sessions:
            if start <= s["start"] < end and in_sleep(s["start"]) and len(s["ibi_ms"]) >= 20:
                ok = clean_ibi(s["ibi_ms"])
                if ok.mean() >= MIN_CLEAN:
                    vals.append(rmssd(s["ibi_ms"], ok))
                    sdnns.append(s["sdnn_ms"])
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            r = float(np.median(vals))
            out["hrv"] = {"source": "Apple Health HRV readings", "n_readings": len(vals), "rmssd_ms": r,
                          "ln_rmssd": float(np.log(r)), "apple_sdnn_ms": float(np.median(sdnns))}
    return out


# -- recovery score across nights --------------------------------------------------------
#
# There is no ground truth for "recovery", so this is a transparent formula, not a
# trained model: each measure tonight vs your own previous nights (a personal baseline,
# the approach common in sports science, e.g. Plews et al. 2013 on rolling ln RMSSD).

BASELINE_NIGHTS = 28    # nights of history used as the baseline
MIN_HISTORY = 7         # earlier nights a measure needs before it counts
Z_CLIP = 3.0
WEIGHTS = {"hrv": 0.5, "rhr": 0.3, "sleep": 0.2}
SD_FLOOR = {"hrv": 0.05, "rhr": 1.0, "sleep": 20.0}  # ln ms, bpm, minutes: no alarms on flat history
BANDS = ((35, "low"), (65, "normal"), (101, "high"))


def recovery_scores(nights: list[dict]) -> list[dict]:
    """Score each night against the nights before it (never later ones).

    nights: dicts with "night" (YYYY-MM-DD) and any of "hrv_rmssd_ms", "resting_hr",
    "tst_min" (None/"" when missing). Returns one dict per night, in date order, with
    "recovery_score" (0-100, 50 = normal for you; None until there is enough history),
    "band", per-measure z-scores and a plain-language "note"."""
    rows = sorted(nights, key=lambda r: r["night"])
    vals = {
        "hrv": [_num(r.get("hrv_rmssd_ms"), log=True) for r in rows],
        "rhr": [_num(r.get("resting_hr")) for r in rows],
        "sleep": [_num(r.get("tst_min")) for r in rows],
    }
    out = []
    for i, r in enumerate(rows):
        z, notes = {}, []
        for m, series in vals.items():
            today = series[i]
            hist = [v for v in series[max(0, i - BASELINE_NIGHTS):i] if v is not None]
            if today is None or len(hist) < MIN_HISTORY:
                continue
            mean = float(np.mean(hist))
            sd = max(float(np.std(hist, ddof=1)), SD_FLOOR[m])
            zz = (today - mean) / sd
            if m == "rhr":
                zz = -zz  # lower resting heart rate = better recovered
            z[m] = float(np.clip(zz, -Z_CLIP, Z_CLIP))
            notes.append(_describe(m, today, mean))
        score = band = None
        if "hrv" in z or "rhr" in z:   # sleep length alone is not a recovery signal
            w = {m: WEIGHTS[m] for m in z}
            wz = sum(w[m] * z[m] for m in z) / sum(w.values())
            score = round(float(np.clip(50 + wz * 50 / Z_CLIP, 0, 100)), 1)
            band = next(name for lim, name in BANDS if score < lim)
        out.append({"night": r["night"], "recovery_score": score, "band": band,
                    **{f"{m}_z": round(z[m], 2) if m in z else None for m in WEIGHTS},
                    "note": "; ".join(notes) if notes else
                    f"needs {MIN_HISTORY} earlier nights with HRV or resting heart rate"})
    return out


def _num(x, log: bool = False):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v) or (log and v <= 0):
        return None
    return float(np.log(v)) if log else v


def _describe(m: str, today: float, mean: float) -> str:
    if m == "hrv":
        pct = 100 * (np.exp(today - mean) - 1)
        return f"HRV {abs(pct):.0f}% {'above' if pct >= 0 else 'below'} your average"
    if m == "rhr":
        return f"resting HR {today - mean:+.1f} bpm vs your average"
    return f"sleep {today - mean:+.0f} min vs your average"
