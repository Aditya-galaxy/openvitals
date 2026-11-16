"""Validate an exported model through the release path, on Walch (lab PSG, 31 subjects).

    uv run python scripts/validate_release.py build/val_ens10_bidsleep253.onnx \
        --walch-raw data/walch --walch-features feat [--out docs/release_validation.md]

1. ONNX vs benchmark: the model on the stored Walch features, scored on the
   benchmark's epoch mask. For a model trained like a benchmark entry, this must
   reproduce that entry's kappa.
2. Raw path: Walch raw heart rate -> night_from_samples -> Stager, as `openvitals
   stage` runs it (scored where the model had heart-rate data).
3. Robustness: the raw path with heart rate thinned to one sample per 30/60/300 s.

Only heart_rate/ and labels/ are needed from Walch: `openvitals download walch
--include 'heart_rate/*,labels/*'`. Use a model that never saw Walch (e.g. trained
on BIDSleep only), or the numbers are training-set numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openvitals import featstore, metrics, stages
from openvitals.datasets import walch
from openvitals.datasets._io import read_numeric
from openvitals.infer import Stager, night_from_samples

T0 = 1_700_000_000.0  # arbitrary Unix time for epoch 0


def score(cms: dict[str, np.ndarray], classes: list[str]) -> dict:
    stack = np.array([cms[s] for s in sorted(cms)])
    m = metrics.metrics_from_confusion(stack.sum(0), classes)
    ci = metrics.subject_bootstrap(stack, classes, n_boot=2000)
    return {"kappa": m["kappa"], "ci": ci["kappa"], "accuracy": m["accuracy"],
            "sens": {c: m["per_class"][c]["sensitivity"] for c in classes}, "n_epochs": m["n_epochs"]}


def thin(t: np.ndarray, bpm: np.ndarray, every: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first sample in each `every`-second bin, like a device logging at that rate."""
    if every <= 0:
        return t, bpm
    keep = np.r_[True, np.diff(np.floor(t / every)) > 0]
    return t[keep], bpm[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--walch-raw", type=Path, default=Path("data/walch"))
    ap.add_argument("--walch-features", type=Path, default=Path("feat"))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    st = Stager(a.model)
    k = len(st.classes)
    rows = []

    # 1. stored features, benchmark mask
    cms = {}
    for n in featstore.load_dataset(a.walch_features, "walch"):
        from openvitals import features
        probs, _ = st.predict_proba(n)
        y = stages.to_scheme(n.labels, st.scheme)
        m = (y >= 0) & features.extract(n, "motion").scorable
        cms[n.subject_id] = metrics.confusion(y[m], probs.argmax(1)[m], k)
    rows.append(("stored features, benchmark mask", score(cms, st.classes)))

    # 2-3. raw heart rate through night_from_samples -> Stager
    for every in (0, 30, 60, 300):
        cms = {}
        for sid in walch.subject_ids(a.walch_raw):
            hr = read_numeric(a.walch_raw / "heart_rate" / f"{sid}_heartrate.txt", 2)
            labels = walch.labels_to_epochs(read_numeric(a.walch_raw / "labels" / f"{sid}_labeled_sleep.txt", 2))
            t, bpm = thin(hr[:, 0], hr[:, 1], every)
            sel = (t >= 0) & (t < len(labels) * 30)
            if sel.sum() < 10:
                continue
            res = st.stage(night_from_samples(t[sel] + T0, bpm[sel], start=T0, end=T0 + 30 * len(labels)), trim=False)
            y = stages.to_scheme(labels, st.scheme)
            m = (y >= 0) & (res.stage >= 0)
            cms[sid] = metrics.confusion(y[m], res.stage[m].astype(int), k)
        rows.append((f"raw HR, {'native (~5 s)' if not every else f'1 sample / {every} s'}", score(cms, st.classes)))

    head = (f"| path | kappa [95% CI] | accuracy | " + " | ".join(f"{c} sens" for c in st.classes) + " | epochs |\n"
            "|---|---|---|" + "---|" * len(st.classes) + "---|")
    lines = [head] + [f"| {name} | {r['kappa']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {100 * r['accuracy']:.1f}% | "
                      + " | ".join(f"{100 * r['sens'][c]:.0f}%" for c in st.classes) + f" | {r['n_epochs']} |"
                      for name, r in rows]
    report = (f"Model `{a.model.name}` ({st.meta.get('model')}, members {st.meta.get('members', '-')}), "
              f"tested on Walch, 31 subjects.\n\n" + "\n".join(lines) + "\n")
    print(report)
    if a.out:
        a.out.write_text(report)


if __name__ == "__main__":
    main()
