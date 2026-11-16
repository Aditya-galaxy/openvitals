import json
import sqlite3
import zipfile
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from openvitals import adapters, datasets, stages
from openvitals.infer import Stager, night_from_samples

IST = ZoneInfo("Asia/Kolkata")


# -- adapters -------------------------------------------------------------------

def test_csv_iso_and_unix(tmp_path):
    p = tmp_path / "a.csv"
    later = int(datetime(2026, 11, 10, 23, 0, 10, tzinfo=IST).timestamp())  # Unix seconds form
    p.write_text(f"Time,HR\n2026-11-10T23:00:00+05:30,61\n2026-11-10T23:00:05+05:30,0\n{later},63\n")
    t, hr = adapters.read_csv(p)
    assert hr.tolist() == [61, 63]  # 0 bpm dropped
    assert t[0] == datetime(2026, 11, 10, 23, 0, tzinfo=IST).timestamp()
    (tmp_path / "b.csv").write_text("time,bpm\n2026-11-10 23:00:00,60\n")
    with pytest.raises(ValueError, match="no timezone"):
        adapters.read_csv(tmp_path / "b.csv")


def _apple_xml(records):
    rows = "\n".join(
        f'<Record type="{typ}" sourceName="{src}" unit="count/min" startDate="{d}" endDate="{d}" value="{v}"/>'
        for typ, src, d, v in records)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<HealthData locale="en_IN">\n{rows}\n</HealthData>\n'


def test_apple_health_zip_xml_and_source_filter(tmp_path):
    hr_type = "HKQuantityTypeIdentifierHeartRate"
    xml = _apple_xml([
        (hr_type, "Aditya's Apple Watch", "2026-11-10 23:00:00 +0530", 58),
        ("HKQuantityTypeIdentifierStepCount", "iPhone", "2026-11-10 23:00:01 +0530", 12),
        (hr_type, "Oura", "2026-11-10 23:00:02 +0530", 60),
        (hr_type, "Aditya's Apple Watch", "2026-11-10 23:00:05 +0530", 57),
    ])
    (tmp_path / "export.xml").write_text(xml)
    with zipfile.ZipFile(tmp_path / "export.zip", "w") as z:
        z.writestr("apple_health_export/export.xml", xml)
    for p in (tmp_path / "export.xml", tmp_path / "export.zip"):
        t, hr = adapters.read_apple_health(p)
        assert hr.tolist() == [58, 60, 57]
    t, hr = adapters.read_apple_health(tmp_path / "export.zip", source="watch")
    assert hr.tolist() == [58, 57] and t[1] - t[0] == 5


def test_gadgetbridge_picks_hr_table_and_handles_ms(tmp_path):
    db = tmp_path / "Gadgetbridge"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE MI_BAND_ACTIVITY_SAMPLE (TIMESTAMP INTEGER, DEVICE_ID INTEGER, HEART_RATE INTEGER)")
    con.execute("CREATE TABLE USER (ID INTEGER, NAME TEXT)")
    con.execute("CREATE TABLE OTHER_SAMPLE (TIMESTAMP INTEGER, HEART_RATE INTEGER)")
    con.executemany("INSERT INTO MI_BAND_ACTIVITY_SAMPLE VALUES (?,?,?)",
                    [(1762795800 + 60 * i, 1 if i < 5 else 2, hr) for i, hr in enumerate([60, 255, 0, 61, 62, 70, 71])])
    con.execute("INSERT INTO OTHER_SAMPLE VALUES (1762795800000, 65)")
    con.commit()
    con.close()
    assert adapters.gadgetbridge_tables(db) == {"MI_BAND_ACTIVITY_SAMPLE": 5, "OTHER_SAMPLE": 1}
    t, hr = adapters.read_gadgetbridge(db)
    assert hr.tolist() == [60, 61, 62, 70, 71]  # 255 / 0 are "no reading"
    assert adapters.read_gadgetbridge(db, device_id=2)[1].tolist() == [70, 71]
    t, hr = adapters.read_gadgetbridge(db, table="OTHER_SAMPLE")
    assert t.tolist() == [1762795800.0]  # milliseconds converted


def test_split_nights_local_windows():
    t = []
    for day in (10, 11, 13):  # no data on the night of the 12th
        start = datetime(2026, 11, day, 23, 0, tzinfo=IST).timestamp()
        t += list(start + np.arange(0, 7 * 3600, 30))
    nights = adapters.split_nights(np.array(t), tz="Asia/Kolkata")
    assert [str(d) for d, _, _ in nights] == ["2026-11-10", "2026-11-11", "2026-11-13"]
    d, s, e = nights[0]
    assert datetime.fromtimestamp(s, IST).hour == 20 and e - s == 16 * 3600


# -- inference ------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_path(walch_root, tmp_path_factory):
    pytest.importorskip("torch")
    from openvitals.models.sequence import SequenceModel

    nights = datasets.load("walch", walch_root)
    m = SequenceModel("tcn", "hr", max_epochs=15).fit(nights, "4class")
    return m.export_onnx(tmp_path_factory.mktemp("m") / "tcn_hr.onnx")


def _evening_padded_samples(walch_root, interval=5.0):
    """A labelled synthetic night as raw Unix-time HR, with 3 h of awake-level HR before it."""
    n = datasets.load("walch", walch_root)[0]
    t0 = datetime(2026, 11, 10, 23, 0, tzinfo=IST).timestamp()
    hr = n.hr[n.hr[:, 0] >= 0]
    pre_t = np.arange(-3 * 3600, 0, interval)
    t = np.r_[pre_t, hr[:, 0]] + t0
    bpm = np.r_[np.full(len(pre_t), 78.0) + np.random.default_rng(0).normal(0, 4, len(pre_t)), hr[:, 1]]
    if interval > 5:
        keep = np.r_[True, np.diff(np.floor(t / interval)) > 0]
        t, bpm = t[keep], bpm[keep]
    return n, t0, t, bpm


def test_stager_trims_evening_and_scores(walch_root, model_path):
    n, t0, t, bpm = _evening_padded_samples(walch_root)
    night = night_from_samples(t, bpm)
    assert night.meta["unix_start"] < t0 - 2.9 * 3600 and (night.labels == stages.UNKNOWN).all()
    st = Stager(model_path)
    full = st.stage(night, trim=False)
    res = st.stage(night)
    assert len(res.stage) < len(full.stage)                    # evening trimmed away
    assert res.unix_start >= t0 - 3 * 3600 + 2 * 3600          # starts well after the padding begins
    assert res.probs.shape == (len(res.stage), 4) and np.allclose(res.probs.sum(1), 1, atol=1e-4)
    assert set(res.summary) >= {"tst_min", "waso_min", "se", "wake_min", "deep_min"}
    assert res.warnings == []
    # The synthetic stages are learnable: staging the original window recovers them well.
    from openvitals import metrics
    y = stages.to_scheme(n.labels, "4class")
    pred = st.stage(night_from_samples(n.hr[:, 0] + t0, n.hr[:, 1], start=t0, end=t0 + 30 * n.n_epochs),
                    trim=False).stage
    m = (y >= 0) & (pred >= 0)
    assert metrics.cohen_kappa(metrics.confusion(y[m], pred[m], 4)) > 0.5


def test_stager_outputs_and_warnings(walch_root, model_path, tmp_path):
    _, _, t, bpm = _evening_padded_samples(walch_root, interval=60.0)
    res = Stager(model_path).stage(night_from_samples(t, bpm))
    assert any("sampled every ~60 s" in w for w in res.warnings)
    csv = res.to_csv(tmp_path / "n.csv", tz=IST).read_text().splitlines()
    assert csv[0] == "time,stage,p_wake,p_light,p_deep,p_rem" and len(csv) == len(res.stage) + 1
    assert "+05:30" in csv[1]
    d = res.to_dict(tz=IST)
    json.dumps(d)  # serialisable
    hyp = res.hypnogram().splitlines()
    assert [line.split("|")[0].strip() for line in hyp] == ["wake", "rem", "light", "deep"]
    assert len({len(line) for line in hyp}) == 1


def test_cli_stage_csv_end_to_end(walch_root, model_path, tmp_path, capsys):
    from openvitals import cli

    _, _, t, bpm = _evening_padded_samples(walch_root)
    lines = ["time,hr"] + [f"{datetime.fromtimestamp(x, IST).isoformat()},{b:.1f}" for x, b in zip(t, bpm)]
    (tmp_path / "hr.csv").write_text("\n".join(lines) + "\n")
    out = tmp_path / "out"
    assert cli.main(["stage", "--csv", str(tmp_path / "hr.csv"), "--tz", "Asia/Kolkata",
                     "--model", str(model_path), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "2026-11-10" in printed and "asleep" in printed and "deep |" in printed
    assert (out / "2026-11-10.csv").exists() and (out / "2026-11-10.json").exists()
    summary = (out / "nights.csv").read_text().splitlines()
    assert summary[0].startswith("night,start,end,") and summary[1].startswith("2026-11-10,")
    with pytest.raises(SystemExit, match="no night"):
        cli.main(["stage", "--csv", str(tmp_path / "hr.csv"), "--tz", "Asia/Kolkata", "--night", "2026-11-01",
                  "--model", str(model_path)])


def test_minute_level_hr_covers_every_epoch_but_real_gaps_do_not(walch_root, model_path):
    from openvitals.infer import _near_hr

    t0 = datetime(2026, 11, 10, 23, 0, tzinfo=IST).timestamp()
    t = t0 + np.r_[np.arange(0, 3 * 3600, 60), np.arange(4 * 3600, 6 * 3600, 60)]  # 1/min, 1 h gap
    night = night_from_samples(t, np.full(len(t), 60.0), start=t0, end=t0 + 6 * 3600)
    cov = _near_hr(night)
    assert cov[: 3 * 120 - 5].all() and cov[4 * 120 + 5:].all()   # every epoch between minute samples
    assert not cov[3 * 120 + 30: 4 * 120 - 30].any()               # the hour with no data stays uncovered
    res = Stager(model_path).stage(night, trim=False)
    assert (res.stage[3 * 120 + 30: 4 * 120 - 30] == stages.UNKNOWN).all()
    assert (res.stage[: 3 * 120 - 5] >= 0).all()
