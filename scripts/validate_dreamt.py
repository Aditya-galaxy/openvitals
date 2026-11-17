"""Held-out sleep-staging test on DREAMT (100 lab PSG nights, Empatica E4 wristband).

    python scripts/validate_dreamt.py ~/dreamt --models release=<bundled> val=val_ens10_bidsleep253.onnx \
        --out dreamt_results [--progress ~/dreamt/progress.log]

Neither model has seen DREAMT (the shipped one trained on BIDSleep + Walch; `val` on
BIDSleep only), so these are true held-out numbers. Runs the `openvitals stage` path:
raw E4 heart rate -> night_from_samples -> Stager (no trimming; scored where the
model had heart-rate data), at the E4's 1 Hz and thinned to 1 sample / 5 s (what the
models trained on) and / 60 s.

Resumable: each subject's confusion matrices go to <out>/subjects/<sid>.json, so it can
run while files are still downloading (with --progress, only verified files are read).
Only aggregate results leave the VM: per-subject confusion matrices, night totals and
the report. Restricted data (DUA 1.5.0): never copy the raw files off the VM.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from openvitals import metrics, stages
from openvitals.datasets import dreamt
from openvitals.infer import Stager, default_model_path, night_from_samples

T0 = 1_700_000_000.0
RATES = {"1 Hz (E4)": 0, "1 / 5 s": 5, "1 / 60 s": 60}


def thin(t, bpm, every):
    if every <= 0:
        return t, bpm
    keep = np.r_[True, np.diff(np.floor(t / every)) > 0]
    return t[keep], bpm[keep]


def done_files(progress: Path | None) -> set[str] | None:
    if progress is None:
        return None
    return {line.split()[2].split("/")[-1].split("_")[0] for line in progress.read_text().splitlines()
            if " ok data_64Hz/" in line}


def run_subject(path: Path, sid: str, models: dict[str, Stager], cache: Path) -> dict:
    n = dreamt.load(path.parent.parent if path.parent.name == dreamt.SUBDIR else path.parent,
                    cache_dir=cache, subjects=[sid])[0]
    labels = n.labels
    out = {"sid": sid, "n_epochs": int(labels.size), "scored": int((labels >= 0).sum()),
           "e4_offset_s": n.meta["e4_offset_s"], "hr_samples": int(len(n.hr)), "results": {}}
    t_all, bpm_all = n.hr[:, 0], n.hr[:, 1]
    for mname, st in models.items():
        k = len(st.classes)
        y = stages.to_scheme(labels, st.scheme)
        for rname, every in RATES.items():
            t, bpm = thin(t_all, bpm_all, every)
            sel = (t >= 0) & (t < labels.size * 30)
            if sel.sum() < 10:
                continue
            res = st.stage(night_from_samples(t[sel] + T0, bpm[sel], start=T0, end=T0 + 30 * labels.size), trim=False)
            m = (y >= 0) & (res.stage >= 0)
            cm = metrics.confusion(y[m], res.stage[m].astype(int), k)
            sleep = st.sleep_classes
            out["results"][f"{mname}|{rname}"] = {
                "cm": cm.tolist(),
                "tst_true_min": float(np.isin(y[m], list(sleep)).sum() / 2),
                "tst_pred_min": float(np.isin(res.stage[m], list(sleep)).sum() / 2),
            }
    return out


def score(cms: np.ndarray, classes: list[str]) -> dict:
    m = metrics.metrics_from_confusion(cms.sum(0), classes)
    ci = metrics.subject_bootstrap(cms, classes, n_boot=2000)
    per = [metrics.cohen_kappa(c) for c in cms]
    return {"kappa": m["kappa"], "ci": ci["kappa"], "accuracy": m["accuracy"], "n_epochs": m["n_epochs"],
            "sens": {c: m["per_class"][c]["sensitivity"] for c in classes},
            "median_subject_kappa": float(np.nanmedian(per))}


def report(subjects: list[dict], models: dict[str, Stager], info: dict[str, dict]) -> str:
    lines = [f"DREAMT held-out test: {len(subjects)} subjects (lab PSG, Empatica E4 heart rate). "
             "Neither model saw DREAMT. Kappa CI = 95% subject bootstrap.\n"]
    for mname, st in models.items():
        classes = st.classes
        lines += [f"### `{mname}` ({st.meta.get('model')}, {st.scheme})\n",
                  "| heart rate | subjects | kappa [95% CI] | median subject κ | accuracy | "
                  + " | ".join(f"{c} sens" for c in classes) + " | TST bias, min [95% LoA] |",
                  "|---|---|---|---|---|" + "---|" * len(classes) + "---|"]
        for rname in RATES:
            key = f"{mname}|{rname}"
            rs = [s["results"][key] for s in subjects if key in s["results"]]
            if not rs:
                continue
            r = score(np.array([x["cm"] for x in rs]), classes)
            ba = metrics.bland_altman(np.array([x["tst_pred_min"] for x in rs]), np.array([x["tst_true_min"] for x in rs]))
            lines.append(f"| {rname} | {len(rs)} | {r['kappa']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | "
                         f"{r['median_subject_kappa']:.3f} | {100 * r['accuracy']:.1f}% | "
                         + " | ".join(f"{100 * r['sens'][c]:.0f}%" for c in classes)
                         + f" | {ba['bias']:+.0f} [{ba['loa_low']:+.0f}, {ba['loa_high']:+.0f}] |")
        # by sleep-apnea severity (AHI), at the training rate
        key = f"{mname}|1 / 5 s"
        groups = {"AHI < 5": (0, 5), "AHI 5–15": (5, 15), "AHI 15–30": (15, 30), "AHI ≥ 30": (30, 1e9)}
        rows = []
        for g, (lo, hi) in groups.items():
            rs = [s["results"][key]["cm"] for s in subjects
                  if key in s["results"] and lo <= info.get(s["sid"], {}).get("AHI", -1) < hi]
            if len(rs) >= 3:
                r = score(np.array(rs), classes)
                rows.append(f"| {g} | {len(rs)} | {r['kappa']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] |")
        if rows:
            lines += ["", "By sleep-apnea severity (1 / 5 s):", "", "| group | subjects | kappa [95% CI] |",
                      "|---|---|---|"] + rows
        lines.append("")
    return "\n".join(lines)


def read_info(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["SID"]] = {"AHI": float(row["AHI"])}
            except (KeyError, ValueError):
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="DREAMT folder (with data_64Hz/)")
    ap.add_argument("--models", nargs="+", default=["release=bundled"], help="name=path.onnx ('bundled' = shipped)")
    ap.add_argument("--out", type=Path, default=Path("dreamt_results"))
    ap.add_argument("--cache", type=Path, default=Path(".cache/dreamt"))
    ap.add_argument("--progress", type=Path, help="progress.log: only use files it lists as verified")
    a = ap.parse_args()

    models = {}
    for spec in a.models:
        name, _, p = spec.partition("=")
        models[name] = Stager(default_model_path() if p == "bundled" else Path(p))
    (a.out / "subjects").mkdir(parents=True, exist_ok=True)
    ok = done_files(a.progress)
    files = {sid: p for sid, p in dreamt._files(a.root).items() if ok is None or sid in ok}
    for sid, path in files.items():
        f = a.out / "subjects" / f"{sid}.json"
        if f.exists():
            continue
        try:
            r = run_subject(path, sid, models, a.cache)
        except Exception as e:  # one bad file shouldn't stop the run
            print(f"{sid}: FAILED {e!r}", flush=True)
            continue
        f.write_text(json.dumps(r))
        k = {key: round(metrics.cohen_kappa(np.array(v["cm"])), 3) for key, v in r["results"].items() if "1 / 5 s" in key}
        print(f"{sid}: {r['scored']} scored epochs, kappa {k}", flush=True)

    subjects = [json.loads(p.read_text()) for p in sorted((a.out / "subjects").glob("S*.json"))]
    text = report(subjects, models, read_info(a.root / "participant_info.csv"))
    (a.out / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
