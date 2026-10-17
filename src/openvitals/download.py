"""Fetch open-access PhysioNet datasets, verified against the project's SHA256SUMS.txt."""

from __future__ import annotations

import fnmatch
import hashlib
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .datasets import REGISTRY

USER_AGENT = "openvitals-downloader"


def download(name: str, dest: Path, include: str | None = None, dry_run: bool = False,
             workers: int = 8) -> Path:
    base = REGISTRY[name].URL
    dest = Path(dest) / name
    dest.mkdir(parents=True, exist_ok=True)
    sums = _get(base + "SHA256SUMS.txt").decode()
    (dest / "SHA256SUMS.txt").write_text(sums)
    entries = [line.split(maxsplit=1) for line in sums.splitlines() if line.strip()]
    entries = [(h, p.strip().lstrip("*")) for h, p in entries]
    if include:
        globs = [g.strip() for g in include.split(",") if g.strip()]
        entries = [(h, p) for h, p in entries if any(fnmatch.fnmatch(p, g) for g in globs)]
    todo = [(h, p) for h, p in entries if not ((dest / p).exists() and _sha256(dest / p) == h)]
    print(f"{len(entries) - len(todo)} of {len(entries)} files already verified", file=sys.stderr)
    if dry_run:
        for _, rel in todo:
            print(rel, file=sys.stderr)
        return dest
    # Fetch from a mirror when one exists; checksums still come from PhysioNet.
    sources = [u for u in (getattr(REGISTRY[name], "MIRROR", None), base) if u]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_any, sources, rel, dest / rel, digest): rel for digest, rel in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            print(f"[{i}/{len(todo)}] {futures[fut]}", file=sys.stderr, flush=True)
    return dest


def _fetch_any(sources: list[str], rel: str, target: Path, digest: str, attempts: int = 4) -> None:
    """Try each source in turn, retrying transient failures (PhysioNet returns
    occasional 502s) with exponential backoff."""
    for attempt in range(attempts):
        for base in sources:
            try:
                return _fetch(base + rel, target, digest)
            except (OSError, urllib.error.URLError) as e:
                err = e
        time.sleep(2 ** attempt)
    raise err


def _fetch(url: str, target: Path, digest: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(_req(url)) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if _sha256(tmp) != digest:
        tmp.unlink()
        raise IOError(f"checksum mismatch for {target.name}")
    tmp.replace(target)


def _req(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _get(url: str) -> bytes:
    with urllib.request.urlopen(_req(url)) as r:
        return r.read()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()
