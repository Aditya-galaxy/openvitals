import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from openvitals import adapters, recovery

IST = ZoneInfo("Asia/Kolkata")


# -- resting heart rate ---------------------------------------------------------

def test_resting_hr_finds_lowest_sleep_window_and_ignores_wake():
    t0 = 1_700_000_000.0
    n = 600  # 5 h
    t = t0 + np.arange(0, n * 30, 5.0)
    bpm = np.full(len(t), 60.0)
    bpm[(t - t0 >= 2 * 3600) & (t - t0 < 2 * 3600 + 600)] = 50.0   # 10 min dip while asleep
    bpm[(t - t0 >= 4 * 3600) & (t - t0 < 4 * 3600 + 600)] = 45.0   # lower, but awake
    asleep = np.ones(n, bool)
    asleep[480:500] = False                                          # 4:00-4:10 awake
    r = recovery.resting_hr(t, bpm, asleep, t0)
    assert r["lowest_hr_5min"] == pytest.approx(50.0)
    assert 2 * 3600 <= r["lowest_hr_at"] - t0 <= 2 * 3600 + 600
    assert 59.0 < r["avg_sleep_hr"] < 60.0
    none = recovery.resting_hr(t, bpm, np.zeros(n, bool), t0)
    assert none["lowest_hr_5min"] is None and none["avg_sleep_hr"] is None
    # Once-a-minute heart rate leaves half the epochs empty but still gives a value.
    m = (t - t0) % 60 == 0
    coarse = recovery.resting_hr(t[m], bpm[m], asleep, t0)
    assert coarse["lowest_hr_5min"] == pytest.approx(50.0)


# -- HRV --------------------------------------------------------------------------

def _clean_series(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    # 1 s mean interval, slow drift plus beat-to-beat variation (RMSSD ~ 40 ms)
    return 1000 + 30 * np.sin(np.arange(n) / 40) + rng.normal(0, 28, n)


def test_clean_ibi_flags_each_artifact_type():
    ibi = _clean_series(300)
    bad = ibi.copy()
    bad[50] = ibi[50] + ibi[51]                     # missed beat: one long interval
    bad[100], bad[101] = ibi[100] * 0.5, ibi[100] * 0.5   # extra beat: two short
    bad[150], bad[151] = ibi[150] * 0.7, ibi[151] * 1.3   # ectopic short-long
    bad[200] = 2500                                  # out of range
    ok = recovery.clean_ibi(bad)
    for i in (50, 100, 101, 150, 151, 200):
        assert not ok[i], i
    assert ok.mean() > 0.95


def test_rmssd_uses_only_successive_clean_pairs():
    ibi = np.array([1000, 1010, 990, 3000, 1000, 1020.0])
    ok = np.array([True, True, True, False, True, True])
    # pairs (0,1), (1,2), (4,5); the removed beat must not pair 2 with 4
    assert recovery.rmssd(ibi, ok) == pytest.approx(np.sqrt((10**2 + 20**2 + 20**2) / 3))
    assert np.isnan(recovery.rmssd(ibi[:2]))


def test_nightly_hrv_windows_and_sleep_filter():
    ibi = _clean_series(3600)                       # ~1 h of beats
    t = np.cumsum(ibi) / 1000
    true = recovery.rmssd(ibi)
    night = recovery.nightly_hrv(t, ibi)
    assert night["n_windows"] >= 10 and night["rmssd_ms"] == pytest.approx(true, rel=0.15)
    assert night["ln_rmssd"] == pytest.approx(np.log(night["rmssd_ms"]))
    first_half = recovery.nightly_hrv(t, ibi, in_sleep=lambda s: s < 1800)
    assert first_half["n_windows"] < night["n_windows"]


# -- readers ----------------------------------------------------------------------

def test_read_ibi_csv_polar_semicolon_and_plain(tmp_path):
    p = tmp_path / "polar.txt"
    p.write_text("Phone timestamp;RR-interval [ms]\n"
                 "2026-11-10T23:00:00.000+05:30;1000\n2026-11-10T23:00:01.000+05:30;1005\n"
                 "2026-11-10T23:00:02.010+05:30;1010\n")
    t, ibi = adapters.read_ibi_csv(p)
    assert ibi.tolist() == [1000, 1005, 1010] and t[1] - t[0] == pytest.approx(1.0)
    q = tmp_path / "rr.csv"
    q.write_text("rr\n0.95\n1.05\n1.00\n")
    t, ibi = adapters.read_ibi_csv(q)
    assert ibi.tolist() == pytest.approx([950, 1050, 1000]) and t.tolist() == pytest.approx([0.95, 2.0, 3.0])


def _hrv_record(start, sdnn, beats):
    kids = "".join(f'<InstantaneousBeatsPerMinute bpm="{b}" time="{tm}"/>' for tm, b in beats)
    body = f"<HeartRateVariabilityMetadataList>{kids}</HeartRateVariabilityMetadataList>" if beats else ""
    return (f'<Record type="HKQuantityTypeIdentifierHeartRateVariabilitySDNN" sourceName="Watch" unit="ms" '
            f'startDate="{start}" endDate="{start}" value="{sdnn}">{body}</Record>')


def test_apple_health_hrv_formats_and_midnight(tmp_path):
    xml = ("<HealthData>"
           + _hrv_record("2026-11-10 23:59:30 +0530", 41.5,
                         [("11:59:30.10 PM", 60), ("11:59:31.10 PM", 61), ("12:00:00.20 AM", 62), ("12:00:01.20 AM", 60)])
           + _hrv_record("2026-11-11 03:10:00 +0530", 55.0,
                         [("03:10:00,50", 55), ("03:10:01,60", 54), ("03:10:02,70", 56)])
           + _hrv_record("2026-11-11 04:00:00 +0530", 50.0, [("garbled", 60), ("x", 61), ("y", 62)])
           + "</HealthData>")
    with zipfile.ZipFile(tmp_path / "export.zip", "w") as z:
        z.writestr("apple_health_export/export.xml", xml)
    out = adapters.read_apple_health_hrv(tmp_path / "export.zip")
    assert out["sdnn"][1].tolist() == [41.5, 55.0, 50.0]           # SDNN kept even when beats unparseable
    assert len(out["sessions"]) == 2
    s0 = out["sessions"][0]
    expect = datetime(2026, 11, 11, 0, 0, 0, 200000, tzinfo=IST).timestamp()
    assert s0["beat_t"][2] == pytest.approx(expect)                 # crossed midnight onto the next day
    assert s0["ibi_ms"][0] == pytest.approx(1000.0)
    assert out["sessions"][1]["ibi_ms"].tolist() == pytest.approx([60000 / 55, 60000 / 54, 60000 / 56])
