"""Fetch open-access PhysioNet datasets, verified against the project's SHA256SUMS.txt."""

from __future__ import annotations

import fnmatch
import hashlib
import http.client
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .datasets import REGISTRY

USER_AGENT = "openvitals-downloader"


def manifest(name: str, dest: Path, refresh: bool = True) -> list[tuple[str, str]]:
    """(sha256, relative path) for every file in the dataset, from SHA256SUMS.txt.
    With refresh=False an existing local copy is reused."""
    local = Path(dest) / name / "SHA256SUMS.txt"
    if refresh or not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(_get(REGISTRY[name].URL + "SHA256SUMS.txt").decode())
    entries = [line.split(maxsplit=1) for line in local.read_text().splitlines() if line.strip()]
    return [(h, p.strip().lstrip("*")) for h, p in entries]


def download(name: str, dest: Path, include: str | None = None, dry_run: bool = False,
             workers: int = 4, paths: list[str] | None = None, refresh_manifest: bool = True,
             quiet_if_done: bool = False) -> Path:
    """include: comma-separated globs; paths: exact relative paths. Either narrows the download."""
    base = REGISTRY[name].URL
    entries = manifest(name, dest, refresh=refresh_manifest)
    dest = Path(dest) / name
    if paths is not None:
        wanted = set(paths)
        entries = [(h, p) for h, p in entries if p in wanted]
    if include:
        globs = [g.strip() for g in include.split(",") if g.strip()]
        entries = [(h, p) for h, p in entries if any(fnmatch.fnmatch(p, g) for g in globs)]
    todo = [(h, p) for h, p in entries if not ((dest / p).exists() and _sha256(dest / p) == h)]
    if not quiet_if_done or todo:
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
            got, secs = fut.result()
            print(f"[{i}/{len(todo)}] {futures[fut]}  {got / 1e6:.1f} MB at {got / 1e6 / max(secs, 1e-3):.2f} MB/s",
                  file=sys.stderr, flush=True)
    return dest


# A read that stalls this long counts as a failure and is retried. Without a timeout
# a server that stops sending without closing the connection hangs the download
# forever, which is how PhysioNet behaves under load.
READ_TIMEOUT = 60
ATTEMPTS = 8


def _fetch_any(sources: list[str], rel: str, target: Path, digest: str,
               attempts: int = ATTEMPTS, timeout: float = READ_TIMEOUT) -> tuple[int, float]:
    """Try each source in turn, retrying failures and stalls with exponential
    backoff (capped at 60 s). Partial files are kept between attempts and resumed."""
    err: Exception | None = None
    for attempt in range(attempts):
        for base in sources:
            try:
                return _fetch(base + rel, target, digest, timeout)
            except (OSError, urllib.error.URLError, http.client.HTTPException) as e:  # timeouts, drops
                err = e
        time.sleep(min(2 ** attempt, 60))
    raise IOError(f"{rel}: gave up after {attempts} attempts: {err}") from err


def _fetch(url: str, target: Path, digest: str, timeout: float = READ_TIMEOUT) -> tuple[int, float]:
    """Download to <target>.part, resuming a partial file with an HTTP Range request,
    then verify the checksum. Returns (bytes this call fetched, seconds)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    req = _req(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    t0, got = time.perf_counter(), 0
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:  # range past the end: the partial file is already complete
            r = None
        else:
            raise
    if r is not None:
        with r:
            resumed = have and r.status == 206
            with open(tmp, "ab" if resumed else "wb") as f:
                try:
                    while chunk := r.read(1 << 16):
                        f.write(chunk)
                        got += len(chunk)
                except http.client.IncompleteRead as e:  # connection dropped mid-chunk:
                    f.write(e.partial)                    # keep what arrived, resume from there
                    raise
            # read(n) returns b"" when the server closes early instead of raising, so check
            # the bytes still owed; the partial file stays for the next attempt to resume.
            if r.length:
                raise http.client.IncompleteRead(b"", r.length)
    if _sha256(tmp) != digest:
        tmp.unlink()  # corrupt or stale partial: start clean next attempt
        raise IOError(f"checksum mismatch for {target.name}")
    tmp.replace(target)
    return got, time.perf_counter() - t0


def _req(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _get(url: str) -> bytes:
    with urllib.request.urlopen(_req(url), timeout=READ_TIMEOUT) as r:
        return r.read()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()
