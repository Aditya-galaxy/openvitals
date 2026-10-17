import json

import pytest

from openvitals import benchmark, cli, datasets, splits


@pytest.fixture(scope="module")
def walch_nights(walch_root):
    return datasets.load("walch", walch_root)


def test_models_beat_majority(walch_nights):
    folds = splits.subject_kfold(walch_nights, k=4)
    report = benchmark.run(["majority", "gbdt_motion_hr", "logreg_motion"], folds, "4class", {
        "split": "kfold", "description": "synthetic"})
    by = {r["model"]: r for r in report["results"]}
    assert by["majority"]["pooled"]["kappa"] == pytest.approx(0.0)
    assert by["gbdt_motion_hr"]["pooled"]["kappa"] > 0.5
    assert by["logreg_motion"]["pooled"]["kappa"] > 0.2
    # Every model is scored on the same epochs.
    assert len({r["pooled"]["n_epochs"] for r in report["results"]}) == 1


def test_cole_kripke_is_sleep_wake_only(walch_nights):
    folds = splits.subject_kfold(walch_nights, k=4)
    four = benchmark.run(["cole_kripke"], folds, "4class", {"split": "kfold", "description": ""})
    assert four["skipped_models"] == ["cole_kripke"]
    two = benchmark.run(["cole_kripke"], folds, "2class", {"split": "kfold", "description": ""})
    wake = two["results"][0]["pooled"]["per_class"]["wake"]["sensitivity"]
    assert wake > 0.5


def test_cli_cross_dataset(walch_root, bidsleep_root, tmp_path, capsys):
    rc = cli.main([
        "bench", f"bidsleep={bidsleep_root}", "--test", f"walch={walch_root}",
        "--split", "cross", "--scheme", "3class", "--models", "majority,gbdt_motion_hr",
        "--out", str(tmp_path / "res"), "--cache", str(tmp_path / "cache"),
    ])
    assert rc == 0
    md = capsys.readouterr().out
    assert "bidsleep_to_walch" in md and "gbdt_motion_hr" in md
    report = json.loads((tmp_path / "res" / "bidsleep_to_walch-3class.json").read_text())
    assert report["setup"]["folds"][0]["n_test"] == 8
    gbdt = next(r for r in report["results"] if r["model"] == "gbdt_motion_hr")
    assert gbdt["pooled"]["kappa"] > 0.3
    assert len(gbdt["nights"]) == 8


def test_hr_only_cross_dataset_without_motion_files(walch_root, bidsleep_root, tmp_path):
    import shutil

    hr_only = tmp_path / "bidsleep_hr"
    shutil.copytree(bidsleep_root, hr_only, ignore=shutil.ignore_patterns("motion.csv"))
    test = datasets.load("bidsleep", hr_only, cache_dir=tmp_path / "c")
    assert all(len(n.motion) == 0 for n in test)
    folds = splits.cross_dataset(datasets.load("walch", walch_root), test)
    report = benchmark.run(["majority", "gbdt_hr"], folds, "3class", {"split": "cross", "description": ""})
    gbdt = next(r for r in report["results"] if r["model"] == "gbdt_hr")
    assert gbdt["pooled"]["n_epochs"] > 0
    assert gbdt["pooled"]["kappa"] > 0.2
