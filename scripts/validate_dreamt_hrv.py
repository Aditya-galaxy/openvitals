"""Wrist HRV vs ECG on DREAMT (data_100Hz: PSG ECG + Empatica E4 BVP on one clock).

    python scripts/validate_dreamt_hrv.py ~/dreamt --out dreamt_hrv [--progress ~/dreamt/progress.log]

For each subject, beats from three sources, all cleaned the same way (recovery.clean_ibi):
  ecg  R-peaks from the PSG ECG (reference)
  ppg  OpenVitals' own beat detection on the wrist BVP waveform (openvitals.beats)
  e4   the E4's own inter-beat intervals (its IBI output; consecutive equal intervals merge,
       which cleaning then drops)
RMSSD per fixed 5-min window (>= 30 beats, >= 80% clean), on a grid starting at the first
scored epoch; a window counts as asleep if >= 80% of it is scored N1/N2/N3/REM.

Also: wrist beat-timing jitter = beat-to-beat variation of the ECG -> pulse delay
(robust SD of successive differences / sqrt 2), the quantity simulated in RECOVERY.md test B2.

Only aggregates leave the VM (per-window RMSSD values and the report); never the raw files
(DREAMT is restricted, DUA 1.5.0). Resumable per subject; writes are atomic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from openvitals import beats, recovery

FS = 100.0
WIN = 300.0
SLEEP = {"N1", "N2", "N3", "R"}
SOURCES = ("ppg", "e4")


def windows(beat_t: np.ndarray, ibi_ms: np.ndarray, edges: np.ndarray) -> list[dict]:
    ok = recovery.clean_ibi(ibi_ms)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (beat_t >= a) & (beat_t < b)
        n = int(m.sum())
        frac = float(ok[m].mean()) if n else 0.0
        valid = n >= 30 and frac >= recovery.MIN_CLEAN
        r = recovery.rmssd(ibi_ms[m], ok[m]) if valid else float("nan")
        hr = float(60000.0 / np.mean(ibi_ms[m][ok[m]])) if valid else float("nan")
        out.append({"n": n, "clean": round(frac, 3), "rmssd": None if not np.isfinite(r) else round(r, 2),
                    "hr": None if not np.isfinite(hr) else round(hr, 2)})
    return out


def pulse_jitter(ecg_t: np.ndarray, ppg_t: np.ndarray, asleep_mask_fn) -> float | None:
    """Robust SD (ms) of beat-to-beat change in ECG -> pulse delay, divided by sqrt 2."""
    j = np.searchsorted(ppg_t, ecg_t + 0.1)
    j = np.clip(j, 0, len(ppg_t) - 1)
    pat = ppg_t[j] - ecg_t
    ok = (pat >= 0.1) & (pat <= 0.6) & asleep_mask_fn(ecg_t)
    rr = np.diff(ecg_t)
    pair = ok[1:] & ok[:-1] & (rr > 0.3) & (rr < 2.0)
    d = np.diff(pat)[pair]
    if d.size < 200:
        return None
    med = np.median(d)
    scale = max(1.4826 * np.median(np.abs(d - med)), 0.010)
    d = d[np.abs(d - med) < 5 * scale]  # drop gross mismatches (missed / extra beats), keep the jitter
    return float(d.std(ddof=1) / np.sqrt(2) * 1000)


def run_subject(path: Path, sid: str) -> dict:
    df = pd.read_csv(path, usecols=["TIMESTAMP", "ECG", "BVP", "IBI", "Sleep_Stage"],
                     dtype={"TIMESTAMP": np.float64, "ECG": np.float32, "BVP": np.float32, "IBI": np.float64,
                            "Sleep_Stage": "category"})
    t = df["TIMESTAMP"].to_numpy()
    stage = df["Sleep_Stage"].astype(str).to_numpy()
    scored = np.flatnonzero(stage != "P")
    t0 = float(t[scored[0]])
    rel = t - t0
    edges = np.arange(0, rel[-1] + WIN, WIN)
    asleep_row = np.isin(stage, list(SLEEP))
    win_idx = np.clip(np.floor(rel / WIN).astype(int), 0, len(edges) - 2)
    asleep_frac = np.bincount(win_idx[rel >= 0], weights=asleep_row[rel >= 0], minlength=len(edges) - 1) / \
        np.maximum(np.bincount(win_idx[rel >= 0], minlength=len(edges) - 1), 1)

    def asleep_at(tt):
        k = np.clip(np.floor(tt / WIN).astype(int), 0, len(asleep_frac) - 1)
        return (tt >= 0) & (asleep_frac[k] >= 0.8)

    src = {}
    ecg_t = beats.ecg_r_peaks(df["ECG"].to_numpy(np.float64), FS) + rel[0]
    src["ecg"] = recovery.ibi_from_beats(ecg_t)
    ppg_t = beats.ppg_beats(df["BVP"].to_numpy(np.float64), FS) + rel[0]
    src["ppg"] = recovery.ibi_from_beats(ppg_t)
    ibi = df["IBI"].to_numpy()
    ch = np.flatnonzero(np.isfinite(ibi) & (np.r_[np.nan, ibi[:-1]] != ibi))
    src["e4"] = (rel[ch], ibi[ch] * 1000.0)

    out = {"sid": sid, "n_windows": len(edges) - 1, "asleep": [bool(f >= 0.8) for f in asleep_frac],
           "beats": {k: int(len(v[0])) for k, v in src.items()},
           "jitter_ms": pulse_jitter(ecg_t, ppg_t, asleep_at)}
    for k, (bt, ib) in src.items():
        out[k] = windows(bt, ib, edges)
    return out


def nightly(s: dict, k: str) -> float | None:
    v = [w["rmssd"] for w, a in zip(s[k], s["asleep"]) if a and w["rmssd"] is not None]
    return float(np.median(v)) if v else None


def report(subjects: list[dict]) -> str:
    L = [f"DREAMT wrist HRV vs ECG: {len(subjects)} subjects, 5-min windows while asleep (PSG).", ""]
    ecg_valid = sum(1 for s in subjects for w, a in zip(s["ecg"], s["asleep"]) if a and w["rmssd"] is not None)
    asleep = sum(sum(s["asleep"]) for s in subjects)
    L.append(f"Sleep windows: {asleep}; with a clean ECG reference: {ecg_valid} ({100 * ecg_valid / max(asleep, 1):.0f}%).")
    L += ["", "### Per 5-minute window (sleep, ECG reference clean)", "",
          "| source | windows with wrist value | median error | 90th pct error | ρ (ln RMSSD) | HR MAE, bpm |",
          "|---|---|---|---|---|---|"]
    for k in SOURCES:
        e, x, y, hr = [], [], [], []
        n_ref = n_both = 0
        for s in subjects:
            for wr, ww, a in zip(s["ecg"], s[k], s["asleep"]):
                if not a or wr["rmssd"] is None:
                    continue
                n_ref += 1
                if ww["rmssd"] is None:
                    continue
                n_both += 1
                e.append(abs(ww["rmssd"] - wr["rmssd"]) / wr["rmssd"])
                x.append(np.log(wr["rmssd"])); y.append(np.log(ww["rmssd"])); hr.append(abs(ww["hr"] - wr["hr"]))
        if not e:
            continue
        rho = stats.spearmanr(x, y).statistic
        L.append(f"| {k} | {n_both} / {n_ref} ({100 * n_both / n_ref:.0f}%) | {100 * np.median(e):.1f}% | "
                 f"{100 * np.percentile(e, 90):.1f}% | {rho:.2f} | {np.median(hr):.2f} |")
    L += ["", "### Nightly value (median of sleep windows), one per subject", "",
          "| source | subjects | median error | 90th pct error | ρ across people | ratio wrist/ECG, geometric mean [95% LoA] |",
          "|---|---|---|---|---|---|"]
    for k in SOURCES:
        pairs = [(nightly(s, "ecg"), nightly(s, k)) for s in subjects]
        pairs = [(a, b) for a, b in pairs if a and b]
        if len(pairs) < 3:
            continue
        a, b = np.array(pairs).T
        e = np.abs(b - a) / a
        d = np.log(b) - np.log(a)
        sd = d.std(ddof=1)
        L.append(f"| {k} | {len(pairs)} | {100 * np.median(e):.1f}% | {100 * np.percentile(e, 90):.1f}% | "
                 f"{stats.spearmanr(a, b).statistic:.2f} | {np.exp(d.mean()):.2f} "
                 f"[{np.exp(d.mean() - 1.96 * sd):.2f}, {np.exp(d.mean() + 1.96 * sd):.2f}] |")
    ref = [nightly(s, "ecg") for s in subjects if nightly(s, "ecg")]
    if ref:
        L.append(f"\nECG nightly RMSSD: median {np.median(ref):.1f} ms (IQR {np.percentile(ref, 25):.1f}–{np.percentile(ref, 75):.1f}).")
    jit = [s["jitter_ms"] for s in subjects if s["jitter_ms"] is not None]
    if jit:
        L.append(f"Wrist pulse timing jitter (ECG → pulse delay, beat to beat): median {np.median(jit):.1f} ms "
                 f"(IQR {np.percentile(jit, 25):.1f}–{np.percentile(jit, 75):.1f}), {len(jit)} subjects.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("dreamt_hrv"))
    ap.add_argument("--progress", type=Path)
    a = ap.parse_args()
    (a.out / "subjects").mkdir(parents=True, exist_ok=True)
    ok = None
    if a.progress:
        ok = {ln.split()[2].split("/")[-1].split("_")[0] for ln in a.progress.read_text().splitlines()
              if " ok data_100Hz/" in ln}
    for p in sorted((a.root / "data_100Hz").glob("S*_PSG_df.csv")):
        sid = p.name.split("_")[0]
        f = a.out / "subjects" / f"{sid}.json"
        if (ok is not None and sid not in ok) or (f.exists() and f.stat().st_size > 0):
            continue
        try:
            r = run_subject(p, sid)
        except Exception as e:
            print(f"{sid}: FAILED {e!r}", flush=True)
            continue
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(r))
        tmp.replace(f)
        print(f"{sid}: beats {r['beats']}, jitter {r['jitter_ms']}", flush=True)
    subjects = [json.loads(p.read_text()) for p in sorted((a.out / "subjects").glob("S*.json"))]
    text = report(subjects)
    (a.out / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
