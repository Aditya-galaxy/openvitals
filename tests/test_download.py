"""Downloader against a local server that stalls, drops connections and ignores Range."""

import hashlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from openvitals import download

DATA = bytes(range(256)) * 4096  # 1 MiB
DIGEST = hashlib.sha256(DATA).hexdigest()


class Handler(BaseHTTPRequestHandler):
    log = []          # (path, range header) per request
    mode = {}         # path -> behaviour for the *first* request

    def log_message(self, *a):
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        Handler.log.append((self.path, rng))
        first = sum(p == self.path for p, _ in Handler.log) == 1
        mode = Handler.mode.get(self.path, "ok") if first else "ok"
        start = int(rng.split("=")[1].rstrip("-")) if rng and "norange" not in self.path else 0
        if start >= len(DATA):
            self.send_response(416); self.end_headers(); return
        body = DATA[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        half = len(body) // 2
        if mode == "stall":      # send half, then go silent with the connection open
            self.wfile.write(body[:half]); self.wfile.flush(); time.sleep(3); return
        if mode == "drop":       # send half, then close
            self.wfile.write(body[:half]); self.wfile.flush(); return
        self.wfile.write(body)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda s: None)  # no backoff waits in tests
    Handler.log, Handler.mode = [], {}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _ranges(path):
    return [r for p, r in Handler.log if p == path]


def test_stalled_read_times_out_and_resumes(server, tmp_path):
    Handler.mode["/stall.bin"] = "stall"
    got, _ = download._fetch_any([server + "/"], "stall.bin", tmp_path / "f", DIGEST, timeout=0.5)
    assert (tmp_path / "f").read_bytes() == DATA
    r = _ranges("/stall.bin")
    assert len(r) == 2 and r[0] is None and r[1].startswith("bytes=") and r[1] != "bytes=0-"
    assert got < len(DATA)  # the second request only fetched the missing half


def test_dropped_connection_resumes(server, tmp_path):
    Handler.mode["/drop.bin"] = "drop"
    download._fetch_any([server + "/"], "drop.bin", tmp_path / "f", DIGEST, timeout=5)
    assert (tmp_path / "f").read_bytes() == DATA
    assert _ranges("/drop.bin")[1] == f"bytes={len(DATA) // 2}-"


def test_server_ignoring_range_restarts_cleanly(server, tmp_path):
    (tmp_path / "f.part").write_bytes(DATA[:1000])
    download._fetch_any([server + "/"], "norange.bin", tmp_path / "f", DIGEST, timeout=5)
    assert (tmp_path / "f").read_bytes() == DATA


def test_complete_partial_and_corrupt_partial(server, tmp_path):
    (tmp_path / "a.part").write_bytes(DATA)  # already complete: server answers 416
    download._fetch_any([server + "/"], "a.bin", tmp_path / "a", DIGEST, timeout=5)
    assert (tmp_path / "a").read_bytes() == DATA
    (tmp_path / "b.part").write_bytes(b"x" * 1000)  # corrupt: resumed bytes fail the checksum
    download._fetch_any([server + "/"], "b.bin", tmp_path / "b", DIGEST, timeout=5)
    assert (tmp_path / "b").read_bytes() == DATA


def test_gives_up_with_clear_error(server, tmp_path):
    with pytest.raises(IOError, match="gave up after 3 attempts"):
        download._fetch_any([server + "/"], "x.bin", tmp_path / "x", "0" * 64, attempts=3, timeout=5)
