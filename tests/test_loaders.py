import numpy as np

from openvitals import datasets, stages
from openvitals.datasets import bidsleep, walch


def test_walch_loads_and_maps_codes(walch_root, tmp_path):
    nights = datasets.load("walch", walch_root, cache_dir=tmp_path)
    assert len(nights) == 8
    n = nights[0]
    assert n.n_epochs == 600
    assert (n.labels[:3] == stages.UNKNOWN).all()
    assert set(np.unique(n.labels[3:])) <= {0, 1, 2, 3, 4}
    assert stages.REM in n.labels  # native 5 -> canonical 4
    assert n.motion.shape[1] == 4 and n.hr.shape[1] == 2
    assert n.motion[0, 0] < 0  # signals may start before PSG


def test_walch_cache_roundtrip(walch_root, tmp_path):
    a = datasets.load("walch", walch_root, cache_dir=tmp_path)[0]
    b = datasets.load("walch", walch_root, cache_dir=tmp_path)[0]
    assert any(tmp_path.glob("walch-*.npz"))
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(a.motion, b.motion)


def test_walch_n4_folds_into_n3():
    rows = np.array([[-30, 0], [0, 0], [30, 4], [90, 5]])
    out = walch.labels_to_epochs(rows)
    assert out.tolist() == [stages.WAKE, stages.N3, stages.UNKNOWN, stages.REM]


def test_bidsleep_aligns_to_rec_start(bidsleep_root):
    nights = datasets.load("bidsleep", bidsleep_root)
    assert len(nights) == 18
    assert {n.subject_id for n in nights} == {f"Bidslab{s:02d}" for s in range(6)}
    n = nights[0]
    # Synthetic signals start 60 s before epoch 0 regardless of DST.
    assert abs(n.motion[0, 0] + 60) < 1e-3
    assert abs(n.hr[0, 0] + 60) < 1e-3
    assert (n.labels[-2:] == stages.UNKNOWN).all()
    assert stages.REM in n.labels


def test_bidsleep_rec_start_parsing():
    dt = bidsleep.parse_rec_start(np.array("2024-07-01 22:00:00"))
    assert dt.utcoffset().total_seconds() == -4 * 3600  # EDT
    dt = bidsleep.parse_rec_start(np.array(739434.5))  # MATLAB datenum
    assert (dt.year, dt.hour) == (2024, 12)


def test_compact_load_keeps_features(walch_root, tmp_path):
    from openvitals import features

    full = datasets.load("walch", walch_root, cache_dir=tmp_path)
    small = datasets.load("walch", walch_root, cache_dir=tmp_path, compact=True)
    assert [n.key for n in full] == [n.key for n in small]
    assert all(len(n.motion) == 0 for n in small)
    for inputs in features.INPUTS:
        a, b = features.extract(full[0], inputs), features.extract(small[0], inputs)
        np.testing.assert_allclose(a.X, b.X)
        assert (a.scorable == b.scorable).all()


def test_compact_feature_cache(walch_root, tmp_path):
    from openvitals import features

    first = datasets.load("walch", walch_root, cache_dir=tmp_path, compact=True)
    assert len(list(tmp_path.glob("features-walch-*.pkl"))) == 8
    second = datasets.load("walch", walch_root, cache_dir=tmp_path, compact=True)
    np.testing.assert_allclose(features.extract(first[0], "motion+hr").X, features.extract(second[0], "motion+hr").X)
    assert second[0].meta["compacted"]


def _write_dreamt(path, rng, n_epochs=200, lead_in=600.0, fs=64):
    """Synthetic DREAMT file: P lead-in, then stages starting at lead_in seconds."""
    from conftest import hypnogram

    hyp = hypnogram(rng, n_epochs)
    names = np.array(["W", "N1", "N2", "N3", "R"])
    t = np.arange(0, lead_in + n_epochs * 30, 1 / fs)
    ep = np.floor((t - lead_in) / 30).astype(int)
    stage = np.where(ep < 0, "P", names[hyp[np.clip(ep, 0, n_epochs - 1)]])
    stage[(ep == 5)] = "Missing"
    hr = np.round(60 + 10 * (stage == "W"), 2)
    ibi = np.where(t < 2, np.nan, 0.8 + 0.1 * (np.floor(t / 0.8) % 2))
    acc = np.column_stack([np.zeros_like(t), np.zeros_like(t), np.full_like(t, 64.0)])
    import pandas as pd

    pd.DataFrame({"TIMESTAMP": t, "BVP": 0.0, "ACC_X": acc[:, 0], "ACC_Y": acc[:, 1], "ACC_Z": acc[:, 2],
                  "TEMP": 34.0, "EDA": 0.1, "HR": hr, "IBI": ibi, "Sleep_Stage": stage,
                  "Obstructive_Apnea": np.nan}).to_csv(path, index=False)
    return hyp


def test_dreamt_aligns_to_first_scored_epoch(tmp_path):
    d = tmp_path / "data_64Hz"
    d.mkdir()
    hyp = _write_dreamt(d / "S002_whole_df.csv", np.random.default_rng(3))
    nights = datasets.load("dreamt", tmp_path, cache_dir=tmp_path / "c")
    n = nights[0]
    assert n.subject_id == "S002" and n.n_epochs == len(hyp)
    assert n.labels[5] == stages.UNKNOWN  # Missing
    keep = np.arange(len(hyp)) != 5
    np.testing.assert_array_equal(n.labels[keep], hyp[keep])  # W,N1,N2,N3,R map 1:1 to canonical
    assert abs(n.meta["e4_offset_s"] - 600) < 1e-6
    assert n.motion[0, 0] == -600 and np.allclose(n.motion[:, 3], 1.0)  # 64 units = 1 g
    assert np.allclose(np.diff(n.hr[:, 0]), 1.0)
    again = datasets.load("dreamt", tmp_path, cache_dir=tmp_path / "c")[0]
    np.testing.assert_array_equal(again.labels, n.labels)
