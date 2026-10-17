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
