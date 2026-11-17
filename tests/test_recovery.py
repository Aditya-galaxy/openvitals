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


# -- recovery score ---------------------------------------------------------------

def _history(n=20, seed=0, start_day=1):
    rng = np.random.default_rng(seed)
    return [{"night": f"2026-10-{d:02d}", "hrv_rmssd_ms": 45 * np.exp(rng.normal(0, 0.1)),
             "resting_hr": 55 + rng.normal(0, 1.5), "tst_min": 440 + rng.normal(0, 30)}
            for d in range(start_day, start_day + n)]


def test_score_normal_nights_near_50_and_stress_night_low():
    nights = _history() + [{"night": "2026-10-21", "hrv_rmssd_ms": 32, "resting_hr": 60, "tst_min": 360}]
    s = recovery.recovery_scores(nights)
    normal = [r["recovery_score"] for r in s[7:-1]]
    assert all(20 < x < 80 for x in normal) and 40 < np.median(normal) < 60
    last = s[-1]
    assert last["recovery_score"] < 20 and last["band"] == "low"
    assert "HRV" in last["note"] and "below" in last["note"] and "resting HR +" in last["note"]


def test_score_needs_history_and_never_uses_future_nights():
    s = recovery.recovery_scores(_history(10))
    assert all(r["recovery_score"] is None for r in s[:7]) and s[7]["recovery_score"] is not None
    assert "needs 7 earlier nights" in s[0]["note"]
    longer = recovery.recovery_scores(_history(10) + _history(10, seed=5, start_day=11))
    assert [r["recovery_score"] for r in longer[:10]] == [r["recovery_score"] for r in s]  # no leakage


def test_score_without_hrv_and_sleep_alone():
    no_hrv = [{**r, "hrv_rmssd_ms": None} for r in _history()]
    s = recovery.recovery_scores(no_hrv)[-1]
    assert s["recovery_score"] is not None and s["hrv_z"] is None and s["rhr_z"] is not None
    sleep_only = [{"night": r["night"], "tst_min": r["tst_min"]} for r in _history()]
    assert recovery.recovery_scores(sleep_only)[-1]["recovery_score"] is None


def test_flat_history_does_not_alarm():
    flat = [{"night": f"2026-10-{d:02d}", "hrv_rmssd_ms": 40, "resting_hr": 55, "tst_min": 450} for d in range(1, 11)]
    flat.append({"night": "2026-10-11", "hrv_rmssd_ms": 41, "resting_hr": 55.5, "tst_min": 440})
    s = recovery.recovery_scores(flat)[-1]
    assert 35 < s["recovery_score"] < 65 and s["band"] == "normal"


def test_trend_cli_merges_files_and_roundtrips_csv(tmp_path, capsys):
    import csv

    from openvitals import cli

    keys = ["night", "tst_min", "resting_hr", "hrv_rmssd_ms"]
    hist = _history(15)
    for name, part in (("a.csv", hist[:10]), ("b.csv", hist[8:])):          # nights 9-10 overlap
        (tmp_path / name).write_text("\n".join([",".join(keys)] + [",".join(str(r[k]) for k in keys) for r in part]) + "\n")
    out = tmp_path / "trend.csv"
    assert cli.main(["trend", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert printed.splitlines()[0].startswith("night") and "2026-10-15" in printed
    rows = list(csv.DictReader(open(out)))
    assert [r["night"] for r in rows] == [r["night"] for r in hist]          # deduplicated, in order
    assert rows[-1]["recovery_score"] and "your average" in rows[-1]["recovery_note"]


def _synthetic_beats(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rr = 1.0 + 0.05 * np.sin(np.arange(n) / 4) + rng.normal(0, 0.02, n)  # ~60 bpm with HRV
    return np.cumsum(rr) + 1.0


def test_ecg_r_peaks_recover_beat_times():
    from openvitals import beats
    fs, t_beats = 100.0, _synthetic_beats()
    t = np.arange(0, t_beats[-1] + 1, 1 / fs)
    ecg = np.zeros_like(t)
    for tb in t_beats:  # narrow R wave + broad T wave, inverted lead
        ecg -= np.exp(-0.5 * ((t - tb) / 0.012) ** 2) + 0.3 * np.exp(-0.5 * ((t - tb - 0.25) / 0.05) ** 2)
    ecg += np.random.default_rng(1).normal(0, 0.02, t.size)
    found = beats.ecg_r_peaks(ecg, fs)
    assert abs(len(found) - len(t_beats)) <= 2
    err = np.abs(found[:, None] - t_beats[None]).min(1)
    assert np.median(err) < 0.004  # sub-sample at 100 Hz


def test_ppg_beats_track_rr_and_rmssd():
    from openvitals import beats, recovery
    fs, t_beats = 64.0, _synthetic_beats(seed=2)
    t = np.arange(0, t_beats[-1] + 1, 1 / fs)
    ppg = np.zeros_like(t)
    for tb in t_beats + 0.25:  # pulse arrives 250 ms after R
        ppg += np.exp(-0.5 * ((t - tb - 0.12) / 0.08) ** 2) + 0.4 * np.exp(-0.5 * ((t - tb - 0.45) / 0.1) ** 2)
    found = beats.ppg_beats(ppg, fs)
    assert abs(len(found) - len(t_beats)) <= 2
    ref = recovery.rmssd(np.diff(t_beats) * 1000)
    est = recovery.rmssd(np.diff(found) * 1000)
    assert abs(est - ref) / ref < 0.10
