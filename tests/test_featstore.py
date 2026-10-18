import shutil

import numpy as np
import pytest

from openvitals import cli, datasets, download, featstore, features


@pytest.fixture
def walch_dir(walch_root, tmp_path):
    d = tmp_path / "data" / "walch"
    shutil.copytree(walch_root, d)
    return d


def test_featurize_local(walch_dir, tmp_path):
    feat = tmp_path / "feat"
    done = featstore.featurize("walch", walch_dir.parent, feat, log=lambda m: None)
    assert len(done) == 8
    assert featstore.featurize("walch", walch_dir.parent, feat, log=lambda m: None) == []  # resumable
    nights = featstore.load_dataset(feat, "walch")
    raw = datasets.load("walch", walch_dir)
    assert [n.key for n in nights] == [n.key for n in raw]
    np.testing.assert_allclose(features.extract(nights[0], "motion+hr").X, features.extract(raw[0], "motion+hr").X)


def test_streaming_download_evicts(walch_root, tmp_path, monkeypatch):
    """--download --evict: fetch one subject, featurize, delete its raw files."""
    rels = sorted(str(p.relative_to(walch_root)) for p in walch_root.rglob("*") if p.is_file())
    monkeypatch.setattr(download, "manifest", lambda name, dest, refresh=True: [("x", r) for r in rels])
    fetched = []

    def fake_fetch(name, dest, paths=None, **kw):
        for rel in paths:
            target = dest / name / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(walch_root / rel, target)
        fetched.append(len(paths))
        return dest / name

    monkeypatch.setattr(download, "download", fake_fetch)
    data_dir, feat = tmp_path / "data", tmp_path / "feat"
    done = featstore.featurize("walch", data_dir, feat, download=True, evict=True, log=lambda m: None)
    assert len(done) == 8 and fetched == [3] * 8  # motion, heart_rate, labels per subject
    assert not any(p.is_file() for p in (data_dir / "walch").rglob("*"))  # all raw evicted
    assert len(featstore.load_dataset(feat, "walch")) == 8


def test_cli_bench_on_feature_store(walch_dir, tmp_path, capsys):
    feat = tmp_path / "feat"
    assert cli.main(["featurize", "walch", "--data-dir", str(walch_dir.parent), "--features", str(feat)]) == 0
    assert cli.main(["bench", f"walch=features:{feat}", "--split", "kfold", "--k", "4", "--scheme", "2class",
                     "--models", "majority,gbdt_motion_hr", "--out", str(tmp_path / "res")]) == 0
    assert "gbdt_motion_hr" in capsys.readouterr().out


def test_version_mismatch_is_refused(walch_dir, tmp_path, monkeypatch):
    feat = tmp_path / "feat"
    featstore.featurize("walch", walch_dir.parent, feat, subjects=["1000"], log=lambda m: None)
    monkeypatch.setattr(features, "FEATURE_VERSION", features.FEATURE_VERSION + 1)
    with pytest.raises(ValueError, match="re-featurize"):
        featstore.load_dataset(feat, "walch")


def test_streaming_redoes_subjects_saved_without_motion(walch_root, tmp_path, monkeypatch):
    """An HR-only feature file must not block a later download that has motion."""
    rels = sorted(str(p.relative_to(walch_root)) for p in walch_root.rglob("*") if p.is_file())
    monkeypatch.setattr(download, "manifest", lambda name, dest, refresh=True: [("x", r) for r in rels])

    def fake_fetch(name, dest, paths=None, **kw):
        for rel in paths:
            (dest / name / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(walch_root / rel, dest / name / rel)

    monkeypatch.setattr(download, "download", fake_fetch)
    feat = tmp_path / "feat"
    hr_only = datasets.load("walch", walch_root, subjects=["1000"])
    hr_only[0].motion = np.empty((0, 4))
    featstore.save_subject(feat, "walch", "1000", hr_only)
    assert not featstore.has_motion(featstore.subject_path(feat, "walch", "1000"))

    done = featstore.featurize("walch", tmp_path / "data", feat, download=True, log=lambda m: None)
    assert "1000" in done and len(done) == 8
    assert featstore.has_motion(featstore.subject_path(feat, "walch", "1000"))


def test_max_nights_fetches_and_keeps_first_nights(bidsleep_root, tmp_path, monkeypatch):
    rels = sorted(str(p.relative_to(bidsleep_root)) for p in bidsleep_root.rglob("*") if p.is_file())
    monkeypatch.setattr(download, "manifest", lambda name, dest, refresh=True: [("x", r) for r in rels])
    fetched = []

    def fake_fetch(name, dest, paths=None, **kw):
        for rel in paths:
            (dest / name / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(bidsleep_root / rel, dest / name / rel)
        fetched.extend(paths)

    monkeypatch.setattr(download, "download", fake_fetch)
    data, feat = tmp_path / "data", tmp_path / "feat"
    leftover = data / "bidsleep" / "Bidslab00" / "2" / "motion.csv"  # from an earlier, interrupted run
    leftover.parent.mkdir(parents=True)
    shutil.copy(bidsleep_root / "Bidslab00" / "2" / "motion.csv", leftover)

    done = featstore.featurize("bidsleep", data, feat, download=True, evict=True, max_nights=1, log=lambda m: None)
    assert len(done) == 6
    assert all(r.split("/")[1] == "1" for r in fetched) and len(fetched) == 6 * 3
    nights = featstore.load_dataset(feat, "bidsleep")
    assert len(nights) == 6 and {n.night_id for n in nights} == {"1"}
    assert all(featstore.has_motion(featstore.subject_path(feat, "bidsleep", s)) for s in done)
    assert not any(p.is_file() for p in (data / "bidsleep").rglob("*") if p.name != "SHA256SUMS.txt")
