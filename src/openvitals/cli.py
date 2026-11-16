"""openvitals command line.

    openvitals download walch --dest data/
    openvitals info walch=data/walch
    openvitals bench walch=data/walch --scheme 4class --split kfold
    openvitals bench bidsleep=data/bidsleep --test walch=data/walch --split cross
    openvitals export walch=data/walch --model tcn_motion_hr --scheme 4class --out tcn.onnx
    openvitals featurize bidsleep --data-dir data --features feat --download --evict
    openvitals bench bidsleep=features:feat --split kfold
    openvitals bench walch=data/walch --models tcn_hr,gbdt_hr --seeds 0,1,2 --reference gbdt_hr
    openvitals merge results/a/kfold-4class-*.json --out results/merged
    openvitals merge results/pooled/kfold-4class-*.json --dataset walch --out results/merged
    openvitals compare results/new.json:gru_hr results/old.json:gru_hr --dataset walch
"""

from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

import numpy as np

from . import benchmark, datasets, featstore, models, splits, stages

DEFAULT_MODELS = list(models.BASELINES)  # sequence models need the seq extra; pass --models


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="openvitals")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download an open-access dataset from PhysioNet")
    d.add_argument("dataset", choices=sorted(datasets.REGISTRY))
    d.add_argument("--dest", type=Path, default=Path("data"))
    d.add_argument("--include", help="comma-separated globs over dataset paths, e.g. '*/hr.csv,*/labels.mat'")
    d.add_argument("--dry-run", action="store_true")

    i = sub.add_parser("info", help="summarise loaded nights")
    i.add_argument("data", nargs="+", metavar="NAME=PATH")
    i.add_argument("--cache", type=Path, default=Path(".cache"))

    b = sub.add_parser("bench", help="run the benchmark")
    b.add_argument("data", nargs="+", metavar="NAME=PATH", help="training data (pooled if several); PATH may be features:<dir>")
    b.add_argument("--test", nargs="*", default=[], metavar="NAME=PATH", help="held-out data for --split cross")
    b.add_argument("--scheme", choices=sorted(stages.SCHEMES), default="4class")
    b.add_argument("--split", choices=["kfold", "loso", "cross"], default="kfold")
    b.add_argument("--k", type=int, default=5)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--models", default=",".join(DEFAULT_MODELS))
    b.add_argument("--out", type=Path, default=Path("results"))
    b.add_argument("--cache", type=Path, default=Path(".cache"))
    b.add_argument("--seeds", default="0", help="comma-separated training seeds, e.g. 0,1,2")
    b.add_argument("--reference", help="model for paired comparisons (default: best pooled kappa)")
    b.add_argument("--bootstrap", type=int, default=benchmark.N_BOOT, help="subject bootstrap draws")
    b.add_argument("--tag", help="suffix for the report file name, e.g. seed1")

    g = sub.add_parser("merge", help="merge reports of one split/scheme (e.g. one per seed)")
    g.add_argument("reports", nargs="+", type=Path)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--reference")
    g.add_argument("--bootstrap", type=int, default=benchmark.N_BOOT)
    g.add_argument("--tag")
    g.add_argument("--dataset", help="score only this dataset's test subjects (e.g. walch of a pooled k-fold)")

    c = sub.add_parser("compare", help="paired Δ kappa of a model in one report against a model in another")
    c.add_argument("a", metavar="REPORT.json:MODEL")
    c.add_argument("b", metavar="REPORT.json:MODEL", help="the reference")
    c.add_argument("--dataset", help="only this dataset's test subjects")
    c.add_argument("--bootstrap", type=int, default=benchmark.N_BOOT)

    e = sub.add_parser("export", help="train a sequence model or ensemble on all given nights and write ONNX")
    e.add_argument("data", nargs="+", metavar="NAME=PATH")
    e.add_argument("--model", choices=sorted({**models.SEQUENCE, **models.ENSEMBLE}), default="tcn_motion_hr")
    e.add_argument("--scheme", choices=sorted(stages.SCHEMES), default="4class")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--jobs", type=int, default=1, help="train ensemble members in parallel processes")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--cache", type=Path, default=Path(".cache"))

    f = sub.add_parser("featurize", help="write per-subject feature files, optionally streaming raw data")
    f.add_argument("dataset", choices=sorted(datasets.REGISTRY))
    f.add_argument("--data-dir", type=Path, default=Path("data"), help="parent of <dataset>/ raw data")
    f.add_argument("--features", type=Path, required=True, help="feature store root")
    f.add_argument("--download", action="store_true", help="fetch each subject's raw files just before use")
    f.add_argument("--evict", action="store_true", help="delete each subject's raw files after featurizing")
    f.add_argument("--subjects", help="comma-separated subset")
    f.add_argument("--workers", type=int, default=4, help="parallel downloads per subject; keep low for PhysioNet")
    f.add_argument("--max-nights", type=int, help="keep each subject's first N nights (multi-night datasets)")
    f.add_argument("--claim", action="store_true", help="share the feature store as a work queue with other processes")
    f.add_argument("--clear-claims", action="store_true", help="remove stale claims first (no other process running)")

    st = sub.add_parser("stage", help="stage your own nights from a device export (no labels needed)")
    src = st.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="CSV with time (ISO 8601 or Unix s) and hr columns")
    src.add_argument("--apple-health", type=Path, help="Apple Health export.zip or export.xml")
    src.add_argument("--gadgetbridge", type=Path, help="Gadgetbridge database export (SQLite)")
    st.add_argument("--source", help="Apple Health: only records whose source name contains this")
    st.add_argument("--table", help="Gadgetbridge: sample table to read (default: most heart-rate rows)")
    st.add_argument("--device-id", type=int, help="Gadgetbridge: DEVICE_ID to keep")
    st.add_argument("--tz", help="IANA timezone for night windows and output (default: this computer's)")
    st.add_argument("--window", default="20:00-12:00", help="local night window, start-end (default 20:00-12:00)")
    st.add_argument("--night", help="only the night starting on this date (YYYY-MM-DD)")
    st.add_argument("--whole", action="store_true", help="treat all samples as one night (no windowing)")
    st.add_argument("--no-trim", action="store_true", help="stage the full window; skip sleep-period detection")
    st.add_argument("--model", type=Path, help="ONNX model (default: bundled HR-only 4-class ensemble)")
    st.add_argument("--out", type=Path, help="write <date>.csv / <date>.json per night and nights.csv here")

    a = p.parse_args(argv)
    if a.cmd == "stage":
        return _stage(a)
    if a.cmd == "merge":
        import json

        merged = benchmark.merge([json.loads(r.read_text()) for r in a.reports], a.reference, a.bootstrap)
        if a.dataset:
            merged = benchmark.restrict(merged, a.dataset, a.reference, a.bootstrap)
        jpath, mpath = benchmark.write(merged, a.out, a.tag)
        print(mpath.read_text())
        print(f"wrote {jpath} and {mpath}", file=sys.stderr)
        return 0
    if a.cmd == "compare":
        import json

        (pa, ma), (pb, mb) = (s.rpartition(":")[::2] for s in (a.a, a.b))
        c = benchmark.compare_reports(json.loads(Path(pa).read_text()), ma, json.loads(Path(pb).read_text()), mb,
                                      a.bootstrap, a.dataset)
        print(f"{a.a} − {a.b} on {c['n_subjects']} subjects: kappa {c['kappa']:.3f} vs {c['reference_kappa']:.3f}, "
              f"Δ {c['delta_kappa']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}], P(Δ ≤ 0) = {c['p_le_0']:.3f}")
        return 0
    if a.cmd == "featurize":
        subjects = [s for s in (a.subjects or "").split(",") if s] or None
        if a.clear_claims:
            featstore.clear_claims(a.features, a.dataset)
        featstore.featurize(a.dataset, a.data_dir, a.features, a.download, a.evict, subjects, a.workers,
                            log=lambda m: print(m, flush=True), max_nights=a.max_nights, claim=a.claim)
        return 0
    if a.cmd == "export":
        nights = _load_all(a.data, a.cache)
        model = models.make(a.model, seed=a.seed)
        if a.model in models.ENSEMBLE:
            model.fit(nights, a.scheme, n_jobs=a.jobs)
        else:
            model.fit(nights, a.scheme)
        print(model.export_onnx(a.out))
        return 0
    if a.cmd == "download":
        from .download import download

        print(download(a.dataset, a.dest, a.include, a.dry_run))
        return 0
    if a.cmd == "info":
        nights = _load_all(a.data, a.cache)
        _print_info(nights)
        return 0

    train = _load_all(a.data, a.cache)
    if a.split == "kfold":
        folds = splits.subject_kfold(train, a.k, a.seed)
    elif a.split == "loso":
        folds = splits.leave_one_subject_out(train)
    else:
        if not a.test:
            p.error("--split cross needs --test NAME=PATH")
        folds = splits.cross_dataset(train, _load_all(a.test, a.cache))
    setup = {
        "split": a.split if a.split != "cross" else folds[0].name.replace("->", "_to_"),
        "k": a.k, "seed": a.seed,
        "description": _describe(train) + (f"; test: {_describe(folds[0].test)}" if a.split == "cross" else ""),
    }
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    report = benchmark.run([m.strip() for m in a.models.split(",") if m.strip()], folds, a.scheme, setup,
                           seeds=seeds, reference=a.reference, n_boot=a.bootstrap)
    jpath, mpath = benchmark.write(report, a.out, a.tag)
    print(mpath.read_text())
    print(f"wrote {jpath} and {mpath}", file=sys.stderr)
    return 0


def _stage(a) -> int:
    import json

    from zoneinfo import ZoneInfo

    from . import adapters
    from .infer import Stager, night_from_samples

    if a.csv:
        t, hr = adapters.read_csv(a.csv)
    elif a.apple_health:
        t, hr = adapters.read_apple_health(a.apple_health, source=a.source)
    else:
        t, hr = adapters.read_gadgetbridge(a.gadgetbridge, table=a.table, device_id=a.device_id)
    if len(t) == 0:
        raise SystemExit("no heart-rate samples found")
    tzname = a.tz or _local_tz()
    tz = ZoneInfo(tzname)
    if a.whole:
        windows = [(datetime.fromtimestamp(t[0], tz).date(), float(t[0]), float(t[-1]) + 1)]
    else:
        ws, we = a.window.split("-")
        windows = adapters.split_nights(t, tz=tzname, window=(ws, we))
        if a.night:
            windows = [w for w in windows if str(w[0]) == a.night]
    if not windows:
        raise SystemExit("no night with at least 3 h of heart-rate data in the window"
                         + (f" on {a.night}" if a.night else "") + "; try --whole or --window")
    stager = Stager(a.model)
    print(f"model: {stager.meta.get('model', stager.path.name)} ({stager.inputs}, {stager.scheme}) | "
          f"{len(t)} HR samples | {len(windows)} night(s) | timezone {tzname}", file=sys.stderr)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d, start, end in windows:
        sel = (t >= start) & (t < end)
        res = stager.stage(night_from_samples(t[sel], hr[sel], start=start, end=end, night_id=str(d)),
                           trim=not a.no_trim)
        info = res.to_dict(tz=tz)
        s = res.summary
        print(f"\n{d}  {info['start'][11:16]}–{info['end'][11:16]}  "
              f"asleep {s.get('tst_min', 0) / 60:.1f} h  efficiency {s.get('se', 0):.0%}  "
              f"onset {s.get('sol_min', 0):.0f} min  WASO {s.get('waso_min', 0):.0f} min  | "
              + "  ".join(f"{c} {s.get(c + '_min', 0):.0f}m" for c in res.classes if c != "wake"))
        print(res.hypnogram())
        for w in res.warnings:
            print(f"  warning: {w}")
        if a.out:
            res.to_csv(a.out / f"{d}.csv", tz=tz)
            (a.out / f"{d}.json").write_text(json.dumps(info, indent=1))
        rows.append({"night": str(d), "start": info["start"], "end": info["end"],
                     **{k: round(v, 3) for k, v in s.items()}, "warnings": " | ".join(res.warnings)})
    if a.out:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        lines = [",".join(keys)] + [",".join(f'"{r.get(k, "")}"' if k == "warnings" else str(r.get(k, ""))
                                             for k in keys) for r in rows]
        (a.out / "nights.csv").write_text("\n".join(lines) + "\n")
        print(f"\nwrote {len(rows)} night(s) to {a.out}/", file=sys.stderr)
    print("\nNot a medical device: estimates for wellness and research use only.", file=sys.stderr)
    return 0


def _local_tz() -> str:
    """This computer's IANA timezone name, from $TZ or /etc/localtime; UTC if unknown."""
    import os

    if os.environ.get("TZ"):
        return os.environ["TZ"]
    real = os.path.realpath("/etc/localtime")
    return real.split("zoneinfo/", 1)[1] if "zoneinfo/" in real else "UTC"


def _load_all(specs: list[str], cache: Path):
    nights = []
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"expected NAME=PATH, got {spec!r}")
        if path.startswith("features:"):
            nights += featstore.load_dataset(Path(path[len("features:"):]), name)
        else:
            nights += datasets.load(name, Path(path), cache_dir=cache, compact=True)
    return nights


def _describe(nights) -> str:
    parts = []
    for ds in sorted({n.dataset for n in nights}):
        sub = [n for n in nights if n.dataset == ds]
        parts.append(f"{ds} ({len({n.subject_id for n in sub})} subjects, {len(sub)} nights)")
    return ", ".join(parts)


def _print_info(nights) -> None:
    print(_describe(nights))
    counts = np.zeros(5, dtype=int)
    unknown = 0
    for n in nights:
        counts += np.bincount(n.labels[n.labels >= 0], minlength=5)[:5]
        unknown += int((n.labels < 0).sum())
    total = counts.sum()
    for code, name in stages.CANONICAL_NAMES.items():
        print(f"  {name:>5}: {counts[code]:7d} epochs ({100 * counts[code] / max(total, 1):.1f}%)")
    print(f"  unscored: {unknown}")


if __name__ == "__main__":
    raise SystemExit(main())
