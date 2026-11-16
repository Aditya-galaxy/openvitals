"""Validate overnight HRV (RMSSD) on ECG beat annotations: nsr2db, 54 adults, ~24 h Holter.

    uv run python scripts/validate_hrv.py --data data/nsr2db [--out build/hrv_validation.md]

Reference ("truth") per subject: median 5-min RMSSD over normal-to-normal (N-N)
intervals, on the 6 h with the lowest mean heart rate (a night proxy: the headers
carry no time of day). Tests:

  A. Real ectopic beats: RMSSD from all beat intervals, raw vs cleaned (clean_ibi).
  B1. Injected artifacts on the N-N series (missed, extra, ectopic beats at 1-5%),
      raw vs cleaned.
  B2. Beat-timing jitter only (5-15 ms SD on the interval, as wrist PPG beat detection
      adds); cleaning cannot remove it, so this bounds PPG-derived RMSSD accuracy.
  C. Apple-Watch-style sampling: k random 1-min readings per night (k = 3, 6, 12),
     intervals from whole-number bpm (Apple stores instantaneous HR as integers).

Errors are |estimate - truth| / truth per subject (median and 90th percentile), plus
Spearman correlation of ln RMSSD across subjects (is the between-person ranking kept?).
Data: https://physionet.org/content/nsr2db/1.0.0/ (ODC-By 1.0); needs the `wfdb` package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from openvitals import recovery

BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
NIGHT_H = 6
RNG = np.random.default_rng(0)


def load(record: Path):
    import wfdb

    a = wfdb.rdann(str(record), "ecg")
    sym = np.array(a.symbol)
    beat = np.isin(sym, list(BEAT_SYMBOLS))
    t = a.sample[beat] / a.fs
    s = sym[beat]
    ibi = np.diff(t) * 1000.0
    nn = (s[1:] == "N") & (s[:-1] == "N")          # interval between two normal beats
    return t[1:], ibi, nn


def night(t, ibi, hours=NIGHT_H):
    """Start of the `hours` window with the lowest mean heart rate (10-min steps)."""
    starts = np.arange(t[0], t[-1] - hours * 3600, 600)
    best, best_hr = starts[0], np.inf
    for s in starts:
        m = (t >= s) & (t < s + hours * 3600)
        if m.sum() > 1000:
            hr = 60000.0 / np.median(ibi[m])
            if hr < best_hr:
                best, best_hr = s, hr
    return (t >= best) & (t < best + hours * 3600)


def nightly(t, ibi, ok):
    """Median 5-min RMSSD using a given clean mask (no further cleaning)."""
    vals = []
    for a in np.arange(t[0], t[-1], recovery.HRV_WINDOW_S):
        m = (t >= a) & (t < a + recovery.HRV_WINDOW_S)
        if m.sum() >= 30 and ok[m].mean() >= recovery.MIN_CLEAN:
            vals.append(recovery.rmssd(ibi[m], ok[m]))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else np.nan


def corrupt(ibi, rate, jitter_ms=0.0):
    """Inject missed beats, extra beats and ectopic short-long pairs at `rate` each third."""
    out = []
    i = 0
    while i < len(ibi):
        r = RNG.random()
        if r < rate / 3 and i + 1 < len(ibi):            # missed beat: merge two intervals
            out.append(ibi[i] + ibi[i + 1]); i += 2; continue
        if r < 2 * rate / 3:                              # extra beat: split one interval
            f = RNG.uniform(0.35, 0.65)
            out += [ibi[i] * f, ibi[i] * (1 - f)]; i += 1; continue
        if r < rate and i + 1 < len(ibi):                 # ectopic: early beat + compensatory pause
            f = RNG.uniform(0.6, 0.8)
            out += [ibi[i] * f, ibi[i + 1] + ibi[i] * (1 - f)]; i += 2; continue
        out.append(ibi[i]); i += 1
    out = np.array(out)
    beats = np.cumsum(out) + RNG.normal(0, jitter_ms / np.sqrt(2), len(out))  # jitter on beat times
    return np.r_[out[0], np.diff(beats)]


def apple_style(t, ibi, nn, k, rounded=True):
    """k random 1-min readings, RMSSD per reading (N-N only), median across readings."""
    starts = RNG.choice(np.arange(t[0], t[-1] - 60, 60), size=k, replace=False)
    vals = []
    for s in starts:
        m = (t >= s) & (t < s + 60) & nn & (ibi >= recovery.IBI_MIN_MS) & (ibi <= recovery.IBI_MAX_MS)
        if m.sum() < 30:
            continue
        x = ibi[m]
        if rounded:
            x = 60000.0 / np.round(60000.0 / x)
        vals.append(recovery.rmssd(x))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else np.nan


def summarise(name, est, truth):
    est, truth = np.asarray(est, float), np.asarray(truth, float)
    ok = np.isfinite(est) & np.isfinite(truth)
    err = np.abs(est[ok] - truth[ok]) / truth[ok]
    rho = spearmanr(np.log(est[ok]), np.log(truth[ok])).statistic
    return f"| {name} | {ok.sum()} | {100 * np.median(err):.1f}% | {100 * np.percentile(err, 90):.1f}% | {rho:.2f} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/nsr2db"))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    recs = sorted({p.with_suffix("") for p in a.data.glob("nsr*.ecg")})
    truth, rows = [], {}
    add = lambda k, v: rows.setdefault(k, []).append(v)  # noqa: E731
    hr = []
    for rec in recs:
        t, ibi, nn = load(rec)
        m = night(t, ibi)
        t, ibi, nn = t[m], ibi[m], nn[m]
        ref = nightly(t, ibi, nn)
        truth.append(ref)
        hr.append(60000.0 / np.median(ibi[nn]))
        # A. real ectopics
        add("A. all beats, no cleaning", nightly(t, ibi, np.ones(len(ibi), bool)))
        add("A. all beats, cleaned", nightly(t, ibi, recovery.clean_ibi(ibi)))
        # B. injected artifacts on the N-N series
        tn = t[nn]
        for rate in (0.01, 0.03, 0.05):
            c = corrupt(ibi[nn], rate, jitter_ms=0.0)
            ct = tn[0] + np.cumsum(c) / 1000.0
            add(f"B1. {rate:.0%} artifacts, no cleaning", nightly(ct, c, np.ones(len(c), bool)))
            add(f"B1. {rate:.0%} artifacts, cleaned", nightly(ct, c, recovery.clean_ibi(c)))
        for jit in (5.0, 10.0, 15.0):
            c = corrupt(ibi[nn], 0.0, jitter_ms=jit)
            ct = tn[0] + np.cumsum(c) / 1000.0
            add(f"B2. {jit:.0f} ms beat-timing jitter, cleaned", nightly(ct, c, recovery.clean_ibi(c)))
        # C. Apple-style sparse 1-min readings
        for k in (3, 6, 12):
            add(f"C. {k} × 1-min readings, exact intervals", apple_style(t, ibi, nn, k, rounded=False))
            add(f"C. {k} × 1-min readings, whole-bpm (Apple)", apple_style(t, ibi, nn, k, rounded=True))
    lines = ["| condition | subjects | median error | 90th pct error | Spearman ρ (ln RMSSD) |",
             "|---|---|---|---|---|"] + [summarise(k, v, truth) for k, v in rows.items()]
    head = (f"nsr2db: {len(recs)} adults, ~24 h Holter ECG, beat annotations manually reviewed. "
            f"Reference nightly RMSSD (N-N, lowest-HR {NIGHT_H} h): median {np.nanmedian(truth):.1f} ms "
            f"(IQR {np.nanpercentile(truth, 25):.1f}–{np.nanpercentile(truth, 75):.1f}); "
            f"night heart rate median {np.median(hr):.0f} bpm.\n\n")
    report = head + "\n".join(lines) + "\n"
    print(report)
    if a.out:
        a.out.write_text(report)


if __name__ == "__main__":
    main()
