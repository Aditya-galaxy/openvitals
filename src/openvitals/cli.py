"""openvitals command line.

    openvitals download walch --dest data/
    openvitals info walch=data/walch
    openvitals bench walch=data/walch --scheme 4class --split kfold
    openvitals bench bidsleep=data/bidsleep --test walch=data/walch --split cross
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import benchmark, datasets, models, splits, stages

DEFAULT_MODELS = list(models.REGISTRY)


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
    b.add_argument("data", nargs="+", metavar="NAME=PATH", help="training data (pooled if several)")
    b.add_argument("--test", nargs="*", default=[], metavar="NAME=PATH", help="held-out data for --split cross")
    b.add_argument("--scheme", choices=sorted(stages.SCHEMES), default="4class")
    b.add_argument("--split", choices=["kfold", "loso", "cross"], default="kfold")
    b.add_argument("--k", type=int, default=5)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--models", default=",".join(DEFAULT_MODELS))
    b.add_argument("--out", type=Path, default=Path("results"))
    b.add_argument("--cache", type=Path, default=Path(".cache"))

    a = p.parse_args(argv)
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
    report = benchmark.run([m.strip() for m in a.models.split(",") if m.strip()], folds, a.scheme, setup)
    jpath, mpath = benchmark.write(report, a.out)
    print(mpath.read_text())
    print(f"wrote {jpath} and {mpath}", file=sys.stderr)
    return 0


def _load_all(specs: list[str], cache: Path):
    nights = []
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"expected NAME=PATH, got {spec!r}")
        nights += datasets.load(name, Path(path), cache_dir=cache)
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
