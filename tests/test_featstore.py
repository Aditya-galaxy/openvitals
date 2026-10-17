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
