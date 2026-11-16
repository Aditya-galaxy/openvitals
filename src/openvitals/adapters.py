"""Read heart rate from real-world exports into (unix_seconds, bpm) arrays.

    csv            columns time (ISO 8601 or Unix seconds) and hr / heart_rate / bpm
    apple_health   Apple Health export.zip or export.xml (streamed; these get large)
    gadgetbridge   Gadgetbridge database export (SQLite); picks the sample table with
                   the most valid heart-rate rows unless one is named

split_nights() cuts a multi-day series into one window per night (local time).
"""

from __future__ import annotations

import csv as _csv
import sqlite3
import zipfile
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HR_MIN, HR_MAX = 25, 230


def read_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no rows")
    cols = {c.lower().strip(): c for c in rows[0]}
    tcol = next((cols[c] for c in ("time", "timestamp", "datetime", "date") if c in cols), None)
    hcol = next((cols[c] for c in ("hr", "heart_rate", "heartrate", "bpm") if c in cols), None)
    if not tcol or not hcol:
        raise ValueError(f"{path}: need a time column and an hr/heart_rate/bpm column, got {list(rows[0])}")
    t = np.array([_parse_time(r[tcol]) for r in rows])
    hr = np.array([float(r[hcol]) if r[hcol] not in ("", None) else np.nan for r in rows])
    return _clean(t, hr)


def read_apple_health(path: Path, source: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Heart-rate records from an Apple Health export. source: keep only records whose
    sourceName contains this text (e.g. "Watch"), useful when several devices log HR."""
    import xml.etree.ElementTree as ET

    ts, hr = [], []
    with _open_export(path) as fh:
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag == "Record" and el.get("type") == "HKQuantityTypeIdentifierHeartRate":
                if source is None or source.lower() in (el.get("sourceName") or "").lower():
                    ts.append(datetime.strptime(el.get("startDate"), "%Y-%m-%d %H:%M:%S %z").timestamp())
                    hr.append(float(el.get("value")))
            if el.tag in ("Record", "Workout", "ActivitySummary"):
                el.clear()  # keep memory flat on multi-GB exports
    return _clean(np.array(ts), np.array(hr))


def _open_export(path: Path):
    """Apple Health export.xml, directly or from inside export.zip (binary file object)."""
    path = Path(path)
    if path.suffix != ".zip":
        return open(path, "rb")
    z = zipfile.ZipFile(path)
    name = next((n for n in z.namelist() if n.endswith("export.xml")), None)
    if name is None:
        raise ValueError(f"{path}: no export.xml inside")
    return z.open(name)


def gadgetbridge_tables(path: Path) -> dict[str, int]:
    """Sample tables with TIMESTAMP and HEART_RATE columns -> count of valid HR rows."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        out = {}
        for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            cols = {r[1].upper() for r in con.execute(f'PRAGMA table_info("{name}")')}
            if {"TIMESTAMP", "HEART_RATE"} <= cols:
                (n,) = con.execute(f'SELECT COUNT(*) FROM "{name}" WHERE HEART_RATE > ? AND HEART_RATE < ?',
                                   (HR_MIN, HR_MAX)).fetchone()
                out[name] = n
        return out
    finally:
        con.close()


def read_gadgetbridge(path: Path, table: str | None = None, device_id: int | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    tables = gadgetbridge_tables(path)
    if not tables:
        raise ValueError(f"{path}: no table with TIMESTAMP and HEART_RATE columns")
    if table is None:
        table = max(tables, key=tables.get)
    elif table not in tables:
        raise ValueError(f"{path}: {table!r} has no heart rate; candidates: {sorted(tables)}")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        q, args = f'SELECT TIMESTAMP, HEART_RATE FROM "{table}"', []
        if device_id is not None:
            q, args = q + " WHERE DEVICE_ID = ?", [device_id]
        rows = np.array(con.execute(q + " ORDER BY TIMESTAMP", args).fetchall(), dtype=float).reshape(-1, 2)
    finally:
        con.close()
    t, hr = rows[:, 0], rows[:, 1]
    if len(t) and np.median(t) > 1e11:  # some devices store milliseconds
        t = t / 1000.0
    return _clean(t, hr)  # 0 and 255 are "no reading" markers; _clean drops them


def split_nights(t: np.ndarray, tz: str = "UTC", window: tuple[str, str] = ("20:00", "12:00"),
                 min_hours: float = 3.0) -> list[tuple[date, float, float]]:
    """One (evening date, start_unix, end_unix) per night with at least min_hours of data
    inside the local-time window (start on the evening date, end on the next morning)."""
    if len(t) == 0:
        return []
    zone = ZoneInfo(tz)
    ws, we = (time.fromisoformat(x) for x in window)
    first = datetime.fromtimestamp(float(np.min(t)), zone).date() - timedelta(days=1)
    last = datetime.fromtimestamp(float(np.max(t)), zone).date()
    out, d = [], first
    while d <= last:
        start = datetime.combine(d, ws, zone).timestamp()
        end = datetime.combine(d + timedelta(days=1 if we <= ws else 0), we, zone).timestamp()
        inside = t[(t >= start) & (t < end)]
        if len(inside) > 1 and (inside.max() - inside.min()) / 3600 >= min_hours:
            out.append((d, start, end))
        d += timedelta(days=1)
    return out


def _parse_time(s: str) -> float:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"timestamp {s!r} has no timezone; use ISO 8601 with an offset or Unix seconds")
    return dt.timestamp()


def _clean(t: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, hr = np.asarray(t, float), np.asarray(hr, float)
    ok = np.isfinite(t) & np.isfinite(hr) & (hr > HR_MIN) & (hr < HR_MAX)
    t, hr = t[ok], hr[ok]
    order = np.argsort(t, kind="stable")
    t, hr = t[order], hr[order]
    keep = np.r_[True, np.diff(t) > 0] if len(t) else np.array([], bool)  # drop duplicate timestamps
    return t[keep], hr[keep]


# -- beat-to-beat data (for HRV) ----------------------------------------------------

def read_ibi_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Beat intervals from a CSV (e.g. Polar Sensor Logger, HRV apps): a column whose name
    contains "rr" or "ibi" (ms, or seconds if values < 3), and optionally a time column
    (ISO 8601 or Unix s). Without times, beats are placed at the running sum of intervals
    from t = 0. Returns (beat time s, interval ms) for each beat."""
    with open(path, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            delimiter = _csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except _csv.Error:  # a single column has no delimiter to detect
            delimiter = ","
        rows = list(_csv.DictReader(f, delimiter=delimiter))
    if not rows:
        raise ValueError(f"{path}: no rows")
    names = list(rows[0])
    icol = next((c for c in names if any(k in c.lower() for k in ("rr", "ibi"))), None)
    if icol is None:
        raise ValueError(f"{path}: need a column named like rr / ibi (beat interval), got {names}")
    tcol = next((c for c in names if c.lower().strip() in ("time", "timestamp", "datetime")
                 or "timestamp" in c.lower()), None)
    ibi = np.array([float(r[icol]) if r[icol] not in ("", None) else np.nan for r in rows])
    if np.nanmedian(ibi) < 3:  # seconds
        ibi = ibi * 1000.0
    if tcol is not None:
        t = np.array([_parse_time(r[tcol]) for r in rows])
    else:
        t = np.nancumsum(ibi) / 1000.0
    ok = np.isfinite(t) & np.isfinite(ibi)
    order = np.argsort(t[ok], kind="stable")
    return t[ok][order], ibi[ok][order]


def read_apple_health_hrv(path: Path, source: str | None = None) -> dict:
    """Apple Health HRV: each HKQuantityTypeIdentifierHeartRateVariabilitySDNN record is a
    ~1 minute reading with Apple's SDNN value and, usually, the instantaneous heart rate
    of each beat (integer bpm, time of day without a date).

    Returns {"sdnn": (unix_t, ms), "beats": (unix_t, ibi_ms), "sessions": [...]} where
    each session lists its beats separately (windows are ~1 min, so RMSSD is computed
    per session, never across sessions). Best effort: Apple publishes no schema for
    export.xml, so time formats are parsed leniently and unparseable series are skipped."""
    import xml.etree.ElementTree as ET

    sdnn_t, sdnn_v, sessions = [], [], []
    with _open_export(path) as fh:
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag == "Record" and el.get("type") == "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":
                if source is None or source.lower() in (el.get("sourceName") or "").lower():
                    start = datetime.strptime(el.get("startDate"), "%Y-%m-%d %H:%M:%S %z")
                    sdnn_t.append(start.timestamp())
                    sdnn_v.append(float(el.get("value")))
                    beats = _apple_beats(el, start)
                    if beats is not None:
                        sessions.append({"start": start.timestamp(), "sdnn_ms": float(el.get("value")),
                                         "beat_t": beats[0], "ibi_ms": beats[1]})
            if el.tag in ("Record", "Workout", "ActivitySummary"):
                el.clear()
    all_t = np.concatenate([s["beat_t"] for s in sessions]) if sessions else np.array([])
    all_i = np.concatenate([s["ibi_ms"] for s in sessions]) if sessions else np.array([])
    return {"sdnn": (np.array(sdnn_t), np.array(sdnn_v)), "beats": (all_t, all_i), "sessions": sessions}


_CLOCK_FORMATS = ("%I:%M:%S.%f %p", "%I:%M:%S %p", "%H:%M:%S.%f", "%H:%M:%S")


def _apple_beats(record, start: datetime) -> tuple[np.ndarray, np.ndarray] | None:
    beats = record.findall(".//InstantaneousBeatsPerMinute")
    if len(beats) < 3:
        return None
    times, bpm = [], []
    for b in beats:
        clock = _parse_clock((b.get("time") or "").replace(",", ".").strip())
        if clock is None or not b.get("bpm"):
            return None
        stamp = start.replace(hour=clock.hour, minute=clock.minute, second=clock.second,
                              microsecond=clock.microsecond)
        if stamp < start - timedelta(hours=12):   # series crossed midnight
            stamp += timedelta(days=1)
        times.append(stamp.timestamp())
        bpm.append(float(b.get("bpm")))
    t, hr = np.array(times), np.array(bpm)
    return t, 60000.0 / hr


def _parse_clock(s: str):
    for fmt in _CLOCK_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None
