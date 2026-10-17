import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score

from openvitals import datasets, metrics, splits


def test_kappa_and_f1_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 2000)
    p = np.where(rng.random(2000) < 0.6, y, rng.integers(0, 4, 2000))
    m = metrics.epoch_metrics(y, p, ["a", "b", "c", "d"])
    assert m["kappa"] == pytest.approx(cohen_kappa_score(y, p))
    assert m["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert m["accuracy"] == pytest.approx((y == p).mean())


def test_sleep_summary():
    # 2 wake, 4 sleep, 2 wake, 2 sleep, 1 wake  (epoch = 0.5 min)
    y = np.array([0, 0, 1, 1, 2, 3, 0, 0, 1, 1, 0])
    s = metrics.sleep_summary(y, {1, 2, 3})
    assert s["tst_min"] == 3.0
    assert s["sol_min"] == 1.0
    assert s["waso_min"] == 1.0  # trailing wake after final sleep is not WASO
    assert s["tib_min"] == 5.5


def test_bland_altman():
    b = metrics.bland_altman([10, 12, 14], [10, 10, 10])
    assert b["bias"] == pytest.approx(2.0)
    assert b["loa_high"] - b["bias"] == pytest.approx(1.96 * 2.0)


def test_subject_kfold_has_no_leakage(bidsleep_root):
    nights = datasets.load("bidsleep", bidsleep_root)
    folds = splits.subject_kfold(nights, k=3, seed=0)
    assert len(folds) == 3
    seen = set()
    for f in folds:
        tr = {n.subject_id for n in f.train}
        te = {n.subject_id for n in f.test}
        assert not tr & te
        seen |= te
        assert len(f.train) + len(f.test) == len(nights)
    assert seen == {n.subject_id for n in nights}


def test_loso(walch_root):
    nights = datasets.load("walch", walch_root)
    folds = splits.leave_one_subject_out(nights)
    assert len(folds) == 8
    assert all(len(f.test) == 1 for f in folds)
