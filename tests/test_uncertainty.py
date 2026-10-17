import json

import numpy as np
import pytest

from openvitals import benchmark, cli, datasets, metrics, splits

CLASSES = ["a", "b", "c", "d"]


def _random_cms(rng, n_subjects, k=4, n=400, acc=0.6):
    cms = np.zeros((n_subjects, k, k), dtype=np.int64)
    for s in range(n_subjects):
        y = rng.integers(0, k, n)
        p = np.where(rng.random(n) < acc, y, rng.integers(0, k, n))
        cms[s] = metrics.confusion(y, p, k)
    return cms


def test_batch_scores_match_scalar_metrics():
    cms = _random_cms(np.random.default_rng(0), 20)
    b = metrics.batch_scores(cms)
    for i, cm in enumerate(cms):
        m = metrics.metrics_from_confusion(cm, CLASSES)
        assert b["kappa"][i] == pytest.approx(metrics.cohen_kappa(cm))
        assert b["accuracy"][i] == pytest.approx(m["accuracy"])
        assert b["macro_f1"][i] == pytest.approx(m["macro_f1"])


def test_metrics_from_confusion_roundtrip():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    assert metrics.metrics_from_confusion(metrics.confusion(y, p, 4), CLASSES) == metrics.epoch_metrics(y, p, CLASSES)


def test_subject_bootstrap_ci_contains_estimate_and_shrinks_with_subjects():
    rng = np.random.default_rng(2)
    small, big = _random_cms(rng, 10), _random_cms(rng, 200)
    for cms in (small, big):
        ci = metrics.subject_bootstrap(cms, CLASSES, n_boot=1000)
        k = metrics.cohen_kappa(cms.sum(0))
        assert ci["kappa"][0] <= k <= ci["kappa"][1]
    width = lambda c: c["kappa"][1] - c["kappa"][0]  # noqa: E731
    assert width(metrics.subject_bootstrap(big, CLASSES, 1000)) < width(metrics.subject_bootstrap(small, CLASSES, 1000))
    assert metrics.subject_bootstrap(small, CLASSES, 500, seed=3) == metrics.subject_bootstrap(small, CLASSES, 500, seed=3)


def test_paired_bootstrap_identical_and_dominant():
    cms = _random_cms(np.random.default_rng(4), 30)
    same = metrics.paired_bootstrap(cms, cms, n_boot=500)
    assert same["delta_kappa"] == 0 and same["ci"] == [0.0, 0.0] and same["p_le_0"] == 1.0
    perfect = np.zeros_like(cms)
    for s in range(len(cms)):
        perfect[s][np.diag_indices(4)] = cms[s].sum(1)
    better = metrics.paired_bootstrap(perfect, cms, n_boot=500)
    assert better["delta_kappa"] > 0.3 and better["p_le_0"] == 0.0


def test_paired_comparison_resolves_small_consistent_gain():
    """Subjects vary a lot (easy vs hard); model a is slightly better on every subject.
    Unpaired CIs overlap, but the paired difference excludes zero."""
    rng = np.random.default_rng(5)
    a, b = [], []
    for s in range(40):
        base = rng.uniform(0.35, 0.85)  # per-subject difficulty
        y = rng.integers(0, 4, 600)
        u = rng.random(600)
        noise = rng.integers(0, 4, 600)
        b.append(metrics.confusion(y, np.where(u < base, y, noise), 4))
        a.append(metrics.confusion(y, np.where(u < base + 0.04, y, noise), 4))
    a, b = np.array(a), np.array(b)
    ci_a = metrics.subject_bootstrap(a, CLASSES, 2000)["kappa"]
    ci_b = metrics.subject_bootstrap(b, CLASSES, 2000)["kappa"]
    assert ci_a[0] < ci_b[1]  # unpaired intervals overlap
    paired = metrics.paired_bootstrap(a, b, n_boot=2000)
    assert paired["ci"][0] > 0 and paired["p_le_0"] < 0.01


@pytest.fixture(scope="module")
def walch_nights(walch_root):
    return datasets.load("walch", walch_root)


def test_run_with_seeds_reports_ci_and_comparisons(walch_nights):
    folds = splits.subject_kfold(walch_nights, k=4)
    report = benchmark.run(["majority", "gbdt_hr", "logreg_hr"], folds, "4class",
                           {"split": "kfold", "description": "synthetic"}, seeds=[0, 1], n_boot=300)
    by = {r["model"]: r for r in report["results"]}
    g = by["gbdt_hr"]
    assert g["seeds"] == [0, 1]
    assert g["seed_kappa"]["sd"] == pytest.approx(0.0)  # deterministic model
    assert g["ci"]["kappa"][0] <= g["pooled"]["kappa"] <= g["ci"]["kappa"][1]
    assert report["reference"] == max(report["results"], key=lambda r: r["pooled"]["kappa"])["model"]
    assert {c["model"] for c in report["comparisons"]} == set(by) - {report["reference"]}
    maj = next(c for c in report["comparisons"] if c["model"] == "majority")
    assert maj["delta_kappa"] < 0 and maj["p_le_0"] == 1.0
    md = benchmark.to_markdown(report)
    assert "Paired comparison" in md and "seed SD" in md


def test_merge_equals_joint_run(walch_nights):
    folds = splits.subject_kfold(walch_nights, k=4)
    setup = {"split": "kfold", "description": "synthetic"}
    joint = benchmark.run(["gbdt_hr", "logreg_hr"], folds, "3class", setup, seeds=[0, 1], n_boot=300)
    parts = [benchmark.run(["gbdt_hr", "logreg_hr"], folds, "3class", setup, seeds=[s], n_boot=300) for s in (0, 1)]
    merged = benchmark.merge(parts + parts[:1], n_boot=300)  # duplicate seed 0 is taken once
    for rj, rm in zip(joint["results"], merged["results"]):
        assert rj["model"] == rm["model"] and rm["seeds"] == [0, 1]
        assert rm["pooled"]["kappa"] == pytest.approx(rj["pooled"]["kappa"])
        assert rm["ci"] == rj["ci"]
    with pytest.raises(ValueError, match="one split/scheme"):
        other = benchmark.run(["gbdt_hr"], folds, "2class", setup, n_boot=100)
        benchmark.merge([parts[0], other])


def test_sequence_seeds_differ(walch_nights):
    pytest.importorskip("torch")
    folds = splits.subject_kfold(walch_nights, k=4)[:1]
    r = benchmark.run(["tcn_hr"], folds, "4class", {"split": "kfold", "description": ""}, seeds=[0, 1], n_boot=100)
    vals = r["results"][0]["seed_kappa"]["values"]
    assert vals[0] != vals[1]


def test_cli_seeds_tag_and_merge(walch_root, tmp_path, capsys):
    out = tmp_path / "res"
    for s in ("0", "1"):
        assert cli.main(["bench", f"walch={walch_root}", "--split", "kfold", "--k", "4", "--scheme", "2class",
                         "--models", "majority,gbdt_hr", "--seeds", s, "--tag", f"s{s}", "--bootstrap", "200",
                         "--out", str(out), "--cache", str(tmp_path / "c")]) == 0
    assert (out / "kfold-2class-s0.json").exists() and (out / "kfold-2class-s1.json").exists()
    assert cli.main(["merge", str(out / "kfold-2class-s0.json"), str(out / "kfold-2class-s1.json"),
                     "--reference", "majority", "--bootstrap", "200", "--out", str(tmp_path / "merged")]) == 0
    rep = json.loads((tmp_path / "merged" / "kfold-2class.json").read_text())
    assert rep["reference"] == "majority"
    assert next(r for r in rep["results"] if r["model"] == "gbdt_hr")["seeds"] == [0, 1]
    assert "Paired comparison against `majority`" in capsys.readouterr().out
