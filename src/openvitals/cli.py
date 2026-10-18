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
"""

from __future__ import annotations

import argparse
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

    e = sub.add_parser("export", help="train a sequence model on all given nights and write ONNX")
    e.add_argument("data", nargs="+", metavar="NAME=PATH")
    e.add_argument("--model", choices=sorted(models.SEQUENCE), default="tcn_motion_hr")
    e.add_argument("--scheme", choices=sorted(stages.SCHEMES), default="4class")
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

    a = p.parse_args(argv)
    if a.cmd == "merge":
        import json

        merged = benchmark.merge([json.loads(r.read_text()) for r in a.reports], a.reference, a.bootstrap)
        jpath, mpath = benchmark.write(merged, a.out, a.tag)
        print(mpath.read_text())
        print(f"wrote {jpath} and {mpath}", file=sys.stderr)
        return 0
    if a.cmd == "featurize":
        subjects = [s for s in (a.subjects or "").split(",") if s] or None
        featstore.featurize(a.dataset, a.data_dir, a.features, a.download, a.evict, subjects, a.workers,
                            log=lambda m: print(m, flush=True))
        return 0
    if a.cmd == "export":
        model = models.make(a.model).fit(_load_all(a.data, a.cache), a.scheme)
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
