"""Validate overnight resting heart rate (lowest 5-min mean asleep, and mean sleeping HR).

    uv run python scripts/validate_rhr.py --model build/val_ens10_bidsleep253.onnx \
        [--walch data/walch] [--bidsleep data/bidsleep] [--out build/rhr_validation.md]

1. Staging sensitivity (Walch, lab PSG, 31 nights): resting HR with the sleep mask from
   PSG labels vs from the model's predicted stages. Use a model that never saw Walch.
   Also with heart rate thinned to one sample per 60 s.
2. Night-to-night repeatability (BIDSleep, 47 subjects x 3-7 nights, sleep from the
   EEG-headband labels): within-subject SD, and ICC(1,1), the share of variance that is
   between people. High ICC = the metric reliably tells people apart.

Needs only heart_rate/ + labels/ (Walch) and */hr.csv + */labels.mat (BIDSleep).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openvitals import recovery, stages
from openvitals.datasets import bidsleep, walch
from openvitals.datasets._io import read_numeric
from openvitals.infer import Stager, night_from_samples

T0 = 1_700_000_000.0
KEYS = ("lowest_hr_5min", "avg_sleep_hr")


def thin(t, bpm, every):
    keep = np.r_[True, np.diff(np.floor(t / every)) > 0] if every else np.ones(len(t), bool)
    return t[keep], bpm[keep]


def agreement(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    sd = d.std(ddof=1)
    return f"{d.mean():+.2f} bpm [{d.mean() - 1.96 * sd:+.2f}, {d.mean() + 1.96 * sd:+.2f}], MAE {np.abs(d).mean():.2f} (n={len(d)})"


def icc1(groups):
    """One-way random-effects ICC(1,1) for unbalanced groups, plus within-subject SD."""
    groups = [np.asarray(g, float)[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g) >= 2]
    n = np.array([len(g) for g in groups])
    N, k = n.sum(), len(groups)
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    k0 = (N - (n ** 2).sum() / N) / (k - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw), np.sqrt(msw), k, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--walch", type=Path, default=Path("data/walch"))
    ap.add_argument("--bidsleep", type=Path, default=Path("data/bidsleep"))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    st = Stager(a.model)
    lines = [f"Model for predicted sleep: `{a.model.name}` ({st.meta.get('model')}).", ""]

    # 1. Walch: PSG mask vs predicted mask
    lines += ["### 1. Staging sensitivity (Walch, 31 lab nights)", "",
              "| heart-rate sampling | metric | PSG-mask value (median) | predicted − PSG mask: bias [95% LoA] |",
              "|---|---|---|---|"]
    for every in (0, 60):
        psg, pred = {k: [] for k in KEYS}, {k: [] for k in KEYS}
        for sid in walch.subject_ids(a.walch):
            hr = read_numeric(a.walch / "heart_rate" / f"{sid}_heartrate.txt", 2)
            labels = walch.labels_to_epochs(read_numeric(a.walch / "labels" / f"{sid}_labeled_sleep.txt", 2))
            t, bpm = thin(hr[:, 0], hr[:, 1], every)
            sel = (t >= 0) & (t < 30 * len(labels))
            t, bpm = t[sel] + T0, bpm[sel]
            res = st.stage(night_from_samples(t, bpm, start=T0, end=T0 + 30 * len(labels)), trim=False)
            r_psg = recovery.resting_hr(t, bpm, labels > stages.WAKE, T0)   # labels: -1 unscored, 0 wake
            r_pred = recovery.resting_hr(t, bpm, res.stage > 0, T0)
            for k in KEYS:
                psg[k].append(r_psg[k] if r_psg[k] is not None else np.nan)
                pred[k].append(r_pred[k] if r_pred[k] is not None else np.nan)
        for k in KEYS:
            lines.append(f"| {'native (~5 s)' if not every else f'1 / {every} s'} | {k} | "
                         f"{np.nanmedian(psg[k]):.1f} bpm | {agreement(pred[k], psg[k])} |")

    # 2. BIDSleep repeatability (label mask)
    per_subject = {k: {} for k in KEYS}
    for n in bidsleep.load(a.bidsleep):
        r = recovery.resting_hr(n.hr[:, 0], n.hr[:, 1], n.labels > stages.WAKE, 0.0)
        for k in KEYS:
            if r[k] is not None:
                per_subject[k].setdefault(n.subject_id, []).append(r[k])
    lines += ["", "### 2. Night-to-night repeatability (BIDSleep, sleep from EEG-headband labels)", "",
              "| metric | subjects | nights | within-subject SD | ICC(1,1) |", "|---|---|---|---|---|"]
    for k in KEYS:
        icc, wsd, ns, nn = icc1(list(per_subject[k].values()))
        lines.append(f"| {k} | {ns} | {nn} | {wsd:.2f} bpm | {icc:.2f} |")
    report = "\n".join(lines) + "\n"
    print(report)
    if a.out:
        a.out.write_text(report)


if __name__ == "__main__":
    main()
