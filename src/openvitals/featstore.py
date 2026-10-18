"""Per-subject feature files: compacted nights (features + labels, no raw signals).

Layout: <root>/<dataset>/<subject_id>.pkl

This is what makes large datasets workable on small or ephemeral machines:
download one subject's raw files, featurize, delete the raw files, move on.
A dataset spec of the form NAME=features:<root> loads from here instead of raw
data. Files are pickles written by this package; only load ones you made.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from . import features
from .schema import Night

FORMAT = 1


def subject_path(root: Path, dataset: str, subject_id: str) -> Path:
    return Path(root) / dataset / f"{subject_id}.pkl"


def save_subject(root: Path, dataset: str, subject_id: str, nights: list[Night]) -> Path:
    for n in nights:
        if not n.meta.get("compacted"):
            features.compact(n)
    path = subject_path(root, dataset, subject_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.part")
    with open(tmp, "wb") as f:
        pickle.dump({"format": FORMAT, "feature_version": features.FEATURE_VERSION,
                     "dataset": dataset, "subject_id": subject_id, "nights": nights}, f)
    tmp.replace(path)  # atomic: a crash never leaves a half-written subject
    return path


def load_subject(path: Path) -> list[Night]:
    with open(path, "rb") as f:
        blob = pickle.load(f)
    if blob.get("format") != FORMAT or blob.get("feature_version") != features.FEATURE_VERSION:
        raise ValueError(f"{path} was written by an incompatible version "
                         f"(format {blob.get('format')}, features v{blob.get('feature_version')}); re-featurize")
    return blob["nights"]


def load_dataset(root: Path, dataset: str) -> list[Night]:
    paths = sorted((Path(root) / dataset).glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"no feature files in {Path(root) / dataset}")
    return [n for p in paths for n in load_subject(p)]


def done_subjects(root: Path, dataset: str) -> set[str]:
    return {p.stem for p in (Path(root) / dataset).glob("*.pkl")}


def has_motion(path: Path) -> bool:
    """Whether a saved subject was featurized with motion data."""
    return any(features.extract(n, "motion").motion_cov.any() for n in load_subject(path))


def _has_motion_files(rels: list[str]) -> bool:
    return any("motion" in rel for rel in rels)


def featurize(name: str, data_dir: Path, feat_root: Path, download: bool = False, evict: bool = False,
              subjects: list[str] | None = None, workers: int = 4, log=print,
              max_nights: int | None = None) -> list[str]:
    """Featurize a dataset one subject at a time, resumably.

    Subjects that already have a feature file are skipped. With download=True
    each subject's raw files are fetched (checksum-verified) just before use;
    with evict=True they are deleted right after, so peak disk is one subject.
    max_nights keeps each subject's first N nights (chronological), which cuts
    the download for multi-night datasets; subjects stay the evaluation unit.
    Returns the subjects featurized in this call.
    """
    from . import datasets
    from .download import download as fetch
    from .download import manifest

    mod = datasets.REGISTRY[name]
    root = Path(data_dir) / name
    if download:
        by_subject: dict[str, list[str]] = {}
        for _, rel in manifest(name, data_dir, refresh=False):
            sid = mod.manifest_subject(rel)
            if sid:
                by_subject.setdefault(sid, []).append(rel)
        all_subjects = sorted(by_subject)
        everything = {sid: list(rels) for sid, rels in by_subject.items()}  # for eviction
        if max_nights:
            by_subject = {sid: _first_nights(mod, rels, max_nights) for sid, rels in by_subject.items()}
    else:
        all_subjects = mod.subject_ids(root)
    done = done_subjects(feat_root, name)
    if download:
        # A subject featurized without motion (e.g. an HR-only local copy) is not
        # done if motion is available to download: redo it.
        done = {s for s in done if not _has_motion_files(by_subject.get(s, []))
                or has_motion(subject_path(feat_root, name, s))}
    todo = [s for s in (subjects or all_subjects) if s not in done]
    log(f"{name}: {len(all_subjects) - len(todo)} of {len(all_subjects)} subjects already featurized")
    finished = []
    for i, sid in enumerate(todo, 1):
        if download:
            fetch(name, data_dir, paths=by_subject[sid], workers=workers,
                  refresh_manifest=False, quiet_if_done=True)
        nights = mod.load(root, cache_dir=None, subjects=[sid])
        if max_nights:
            nights = sorted(nights, key=lambda n: _night_order(n.night_id))[:max_nights]
        save_subject(feat_root, name, sid, nights)
        if evict and download:
            for rel in everything[sid]:  # also leftovers from earlier runs
                (root / rel).unlink(missing_ok=True)
        finished.append(sid)
        log(f"[{i}/{len(todo)}] {name}/{sid}: {len(nights)} nights")
    return finished


def _night_order(night_id: str):
    return (0, int(night_id)) if str(night_id).isdigit() else (1, str(night_id))


def _first_nights(mod, rels: list[str], n: int) -> list[str]:
    night_of = getattr(mod, "manifest_night", None)
    if night_of is None:  # one night per subject (e.g. Walch)
        return rels
    keep = set(sorted({night_of(r) for r in rels if night_of(r)}, key=_night_order)[:n])
    return [r for r in rels if night_of(r) in keep]
