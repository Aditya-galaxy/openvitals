import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from openvitals import benchmark, datasets, features, models, splits  # noqa: E402
from openvitals.models.sequence import SequenceModel  # noqa: E402


@pytest.fixture(scope="module")
def walch_nights(walch_root):
    return datasets.load("walch", walch_root)


@pytest.mark.parametrize("arch", ["tcn", "gru"])
def test_sequence_model_learns(walch_nights, arch):
    folds = splits.subject_kfold(walch_nights, k=4)
    fold = folds[0]
    m = SequenceModel(arch, "motion+hr", max_epochs=20, seed=0).fit(fold.train, "4class")
    ys, ps = [], []
    for n in fold.test:
        y, p = benchmark._scored_epochs(n, m.predict(n), "4class")
        ys.append(y)
        ps.append(p)
    from openvitals import metrics

    kappa = metrics.cohen_kappa(metrics.confusion(np.concatenate(ys), np.concatenate(ps), 4))
    assert kappa > 0.4
    assert m.epochs_trained_ >= 1


def test_registry_names_match_model_names():
    for name in models.SEQUENCE:
        assert models.make(name).name == name


def test_onnx_export_matches_torch(walch_nights, tmp_path):
    m = SequenceModel("tcn", "hr", max_epochs=3).fit(walch_nights[:6], "3class")
    path = m.export_onnx(tmp_path / "tcn.onnx")
    sess = ort.InferenceSession(str(path))
    meta = sess.get_modelmeta().custom_metadata_map
    assert json.loads(meta["classes"]) == ["wake", "nrem", "rem"]
    assert json.loads(meta["features"]) == features.extract(walch_nights[0], "hr").names

    for night in walch_nights[6:8]:  # nights of different length: dynamic time axis
        x = features.extract(night, "hr").X.astype(np.float32)[None]
        onnx_logits = sess.run(None, {"features": x})[0][0]
        np.testing.assert_allclose(onnx_logits, m.predict_logits(night), rtol=1e-4, atol=1e-4)
        assert (onnx_logits.argmax(1) == m.predict(night)).all()


def test_gru_onnx_export(walch_nights, tmp_path):
    m = SequenceModel("gru", "motion+hr", max_epochs=2).fit(walch_nights[:6], "2class")
    sess = ort.InferenceSession(str(m.export_onnx(tmp_path / "gru.onnx")))
    night = walch_nights[7]
    x = features.extract(night, "motion+hr").X.astype(np.float32)[None]
    np.testing.assert_allclose(sess.run(None, {"features": x})[0][0], m.predict_logits(night), rtol=1e-4, atol=1e-4)
