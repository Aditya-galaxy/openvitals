"""Per-subject feature files: compacted nights (features + labels, no raw signals).

Layout: <root>/<dataset>/<subject_id>.pkl

This is what makes large datasets workable on small or ephemeral machines:
download one subject's raw files, featurize, delete the raw files, move on.
A dataset spec of the form NAME=features:<root> loads from here instead of raw
data. Files are pickles written by this package; only load ones you made.
"""

from __future__ import annotations

import os
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
    paths = _feature_files(root, dataset)
    if not paths:
        raise FileNotFoundError(f"no feature files in {Path(root) / dataset}")
    return [n for p in paths for n in load_subject(p)]


def done_subjects(root: Path, dataset: str) -> set[str]:
    return {p.stem for p in _feature_files(root, dataset)}


def _feature_files(root: Path, dataset: str) -> list[Path]:
    """Subject files only: skips hidden files such as macOS '._*' AppleDouble stubs,
    which tar on macOS adds to archives and which aren't pickles."""
    return sorted(p for p in (Path(root) / dataset).glob("*.pkl") if not p.name.startswith("."))


def has_motion(path: Path) -> bool:
    """Whether a saved subject was featurized with motion data."""
    return any(features.extract(n, "motion").motion_cov.any() for n in load_subject(path))


def _has_motion_files(rels: list[str]) -> bool:
    return any("motion" in rel for rel in rels)


def featurize(name: str, data_dir: Path, feat_root: Path, download: bool = False, evict: bool = False,
              subjects: list[str] | None = None, workers: int = 4, log=print,
              max_nights: int | None = None, claim: bool = False) -> list[str]:
    """Featurize a dataset one subject at a time, resumably.

    Subjects that already have a feature file are skipped. With download=True
    each subject's raw files are fetched (checksum-verified) just before use;
    with evict=True they are deleted right after, so peak disk is one subject.
    max_nights keeps each subject's first N nights (chronological), which cuts
    the download for multi-night datasets; subjects stay the evaluation unit.
    A subject whose download fails is skipped and reported at the end (IOError),
    after the others are done. With claim=True several processes can share one
    feature store as a work queue: each subject is taken by whichever process
    first creates its claim file, so all stay busy until the list is empty.
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
    if download:  # subjects with raw or partial files already on disk first, so they resume now
        started = lambda sid: any((root / r).exists() or (root / f"{r}.part").exists()  # noqa: E731
                                  for r in by_subject.get(sid, []))
        todo.sort(key=lambda sid: not started(sid))
    log(f"{name}: {len(all_subjects) - len(todo)} of {len(all_subjects)} subjects already featurized")
    finished, failed = [], []
    for i, sid in enumerate(todo, 1):
        if claim:
            if subject_path(feat_root, name, sid).exists() and (
                    not download or not _has_motion_files(by_subject.get(sid, []))
                    or has_motion(subject_path(feat_root, name, sid))):
                continue  # another process finished it since we started
            if not _claim(feat_root, name, sid):
                continue  # another process is on it
        try:
            if download:
                try:
                    fetch(name, data_dir, paths=by_subject[sid], workers=workers,
                          refresh_manifest=False, quiet_if_done=True)
                except OSError as e:  # one unreachable subject shouldn't stop the rest
                    failed.append(sid)
                    log(f"[{i}/{len(todo)}] {name}/{sid}: FAILED download, skipping for now: {e}")
                    continue
            nights = mod.load(root, cache_dir=None, subjects=[sid])
            if max_nights:
                nights = sorted(nights, key=lambda n: _night_order(n.night_id))[:max_nights]
            save_subject(feat_root, name, sid, nights)
            if evict and download:
                for rel in everything[sid]:  # also leftovers from earlier runs
                    (root / rel).unlink(missing_ok=True)
            finished.append(sid)
            log(f"[{i}/{len(todo)}] {name}/{sid}: {len(nights)} nights")
        finally:
            if claim:
                _release(feat_root, name, sid)
    if failed:
        # Partial downloads are kept, so a re-run resumes them.
        raise IOError(f"{len(failed)} subject(s) failed to download: {', '.join(failed)}")
    return finished


def claims_dir(root: Path, dataset: str) -> Path:
    return Path(root) / dataset / ".claims"


def clear_claims(root: Path, dataset: str) -> None:
    """Remove stale claims, e.g. left by processes killed mid-subject."""
    d = claims_dir(root, dataset)
    if d.is_dir():
        for p in d.iterdir():
            p.unlink(missing_ok=True)


def _claim(root: Path, dataset: str, sid: str) -> bool:
    d = claims_dir(root, dataset)
    d.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(d / sid, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # atomic: one winner
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def _release(root: Path, dataset: str, sid: str) -> None:
    (claims_dir(root, dataset) / sid).unlink(missing_ok=True)


def _night_order(night_id: str):
    return (0, int(night_id)) if str(night_id).isdigit() else (1, str(night_id))


def _first_nights(mod, rels: list[str], n: int) -> list[str]:
    night_of = getattr(mod, "manifest_night", None)
    if night_of is None:  # one night per subject (e.g. Walch)
        return rels
    keep = set(sorted({night_of(r) for r in rels if night_of(r)}, key=_night_order)[:n])
    return [r for r in rels if night_of(r) in keep]
