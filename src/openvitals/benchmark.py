"""Run models over subject-wise folds and score them.

Each (model, seed) run produces a *per-seed record*: one confusion matrix per
test subject plus per-night rows. Everything in a report is derived from those
records, so reports from separate processes (e.g. one per seed) can be merged
exactly with `merge`.

Uncertainty is reported two ways:
- seed spread: SD of pooled kappa across training seeds (model instability)
- subject bootstrap: 95% CI from resampling test subjects (who you happened to test on)
Paired bootstrap differences against a reference model use the same subject
draws for both models, so a small but consistent gain can still be resolved.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from . import __version__, features, metrics, models, stages
from .schema import Night
from .splits import Fold

SUMMARY_FIELDS = ("tst_min", "waso_min", "se", "sol_min")
N_BOOT = 2000


# -- running ---------------------------------------------------------------------

def evaluate_seed(model_name: str, folds: list[Fold], scheme: str, seed: int = 0) -> dict | None:
    """Fit on each fold's train nights with `seed`, predict its test nights.
    Returns a per-seed record, or None if the model does not support `scheme`."""
    if scheme not in models.make(model_name).schemes:
        return None
    k = len(stages.class_names(scheme))
    sleep = stages.sleep_index(scheme)
    t0 = time.perf_counter()
    subject_cm: dict[str, np.ndarray] = {}
    night_rows = []
    for fold in folds:
        model = models.make(model_name, seed=seed).fit(fold.train, scheme)
        for night in fold.test:
            y_true, y_pred = _scored_epochs(night, model.predict(night), scheme)
            if y_true.size == 0:
                continue
            cm = metrics.confusion(y_true, y_pred, k)
            key = f"{night.dataset}/{night.subject_id}"
            subject_cm[key] = subject_cm.get(key, 0) + cm
            st, sp = metrics.sleep_summary(y_true, sleep), metrics.sleep_summary(y_pred, sleep)
            night_rows.append({
                "seed": seed, "night": night.key, "fold": fold.name, "n_epochs": int(y_true.size),
                "kappa": metrics.cohen_kappa(cm),
                **{f"true_{f}": st[f] for f in SUMMARY_FIELDS},
                **{f"pred_{f}": sp[f] for f in SUMMARY_FIELDS},
            })
    if not subject_cm:
        raise ValueError("no scorable test epochs")
    subjects = sorted(subject_cm)
    return {"seed": seed, "seconds": time.perf_counter() - t0, "subjects": subjects,
            "subject_confusions": [subject_cm[s].tolist() for s in subjects], "nights": night_rows}


def _scored_epochs(night: Night, pred: np.ndarray, scheme: str) -> tuple[np.ndarray, np.ndarray]:
    """Epochs that are both labelled and have sensor data. Same mask for every model,
    so results are comparable."""
    y = stages.to_scheme(night.labels, scheme)
    mask = (y >= 0) & features.extract(night, "motion").scorable
    return y[mask].astype(np.int64), np.asarray(pred)[mask].astype(np.int64)


def run(model_names: list[str], folds: list[Fold], scheme: str, setup: dict,
        seeds: list[int] = (0,), reference: str | None = None, n_boot: int = N_BOOT) -> dict:
    per_model: dict[str, list[dict]] = {}
    skipped = []
    for name in model_names:
        records = []
        for seed in seeds:
            rec = evaluate_seed(name, folds, scheme, seed)
            if rec is None:
                break
            records.append(rec)
        if records:
            per_model[name] = records
        else:
            skipped.append(name)
    setup = {**setup, "scheme": scheme, "classes": stages.class_names(scheme),
             "folds": [{"name": f.name, "n_train": len(f.train), "n_test": len(f.test)} for f in folds]}
    return build_report(setup, per_model, skipped, reference, n_boot)


def build_report(setup: dict, per_model: dict[str, list[dict]], skipped: list[str],
                 reference: str | None = None, n_boot: int = N_BOOT) -> dict:
    classes = setup["classes"]
    results = [aggregate(name, setup["scheme"], recs, classes, n_boot) for name, recs in per_model.items()]
    if results and reference is None:
        reference = max(results, key=lambda r: r["pooled"]["kappa"])["model"]
    comparisons = _compare(results, reference, n_boot) if results else []
    return {
        "openvitals_version": __version__,
        "python": platform.python_version(),
        "setup": setup,
        "skipped_models": skipped,
        "reference": reference,
        "results": results,
        "comparisons": comparisons,
    }


def aggregate(model: str, scheme: str, records: list[dict], classes: list[str], n_boot: int = N_BOOT) -> dict:
    """Combine per-seed records. Pooled metrics and CIs use each subject's confusion
    summed over seeds (seeds as repeated measurements of the same subjects)."""
    records = sorted(_dedupe(records), key=lambda r: r["seed"])
    subjects = records[0]["subjects"]
    for r in records[1:]:
        if r["subjects"] != subjects:
            raise ValueError(f"{model}: seeds were scored on different subjects")
    stack = np.array([r["subject_confusions"] for r in records], dtype=np.int64)  # (seeds, S, k, k)
    subj = stack.sum(0)
    pooled = metrics.metrics_from_confusion(subj.sum(0), classes)
    seed_scores = metrics.batch_scores(stack.sum(1))
    subj_scores = metrics.batch_scores(subj)
    night_rows = [row for r in records for row in r["nights"]]
    agreement = {f: _mean_agreement(night_rows, f, [r["seed"] for r in records]) for f in SUMMARY_FIELDS}
    return {
        "model": model, "scheme": scheme,
        "seeds": [r["seed"] for r in records],
        "pooled": pooled,
        "seed_kappa": {"values": seed_scores["kappa"].tolist(), "mean": float(seed_scores["kappa"].mean()),
                       "sd": float(seed_scores["kappa"].std(ddof=1)) if len(records) > 1 else 0.0},
        "ci": metrics.subject_bootstrap(subj, classes, n_boot),
        "per_subject": {
            "n_subjects": len(subjects),
            "kappa_mean": _nanmean(subj_scores["kappa"]), "kappa_sd": _nanstd(subj_scores["kappa"]),
            "accuracy_mean": _nanmean(subj_scores["accuracy"]), "accuracy_sd": _nanstd(subj_scores["accuracy"]),
        },
        "night_agreement": agreement,
        "seconds": float(sum(r["seconds"] for r in records)),
        "per_seed": records,
    }


def _compare(results: list[dict], reference: str, n_boot: int) -> list[dict]:
    ref = next((r for r in results if r["model"] == reference), None)
    if ref is None:
        raise ValueError(f"reference model {reference!r} is not in this report")
    out = []
    for r in results:
        if r["model"] == reference:
            continue
        a, b = _subject_stack(r), _subject_stack(ref)
        if _subjects(r) != _subjects(ref):
            continue
        out.append({"model": r["model"], "reference": reference, **metrics.paired_bootstrap(a, b, n_boot)})
    return out


def _subjects(result: dict) -> list[str]:
    return result["per_seed"][0]["subjects"]


def _subject_stack(result: dict) -> np.ndarray:
    return np.array([r["subject_confusions"] for r in result["per_seed"]], dtype=np.int64).sum(0)


def _mean_agreement(rows: list[dict], field: str, seeds: list[int]) -> dict:
    """Bland-Altman per seed, then averaged, so extra seeds don't shrink the limits."""
    per_seed = [metrics.bland_altman([r[f"pred_{field}"] for r in rows if r["seed"] == s],
                                     [r[f"true_{field}"] for r in rows if r["seed"] == s]) for s in seeds]
    per_seed = [b for b in per_seed if b["n"]]
    if not per_seed:
        return metrics.bland_altman([], [])
    keys = ("bias", "loa_low", "loa_high", "mae")
    return {"n": per_seed[0]["n"], **{k: float(np.mean([b[k] for b in per_seed])) for k in keys}}


def _dedupe(records: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in records:
        if r["seed"] not in seen:
            seen.add(r["seed"])
            out.append(r)
    return out


# -- merging reports ---------------------------------------------------------------

def merge(reports: list[dict], reference: str | None = None, n_boot: int = N_BOOT) -> dict:
    """Merge reports of the same split and scheme (e.g. one process per seed or per
    model). Records for the same model and seed are taken once."""
    if not reports:
        raise ValueError("nothing to merge")
    key = lambda r: (r["setup"]["split"], r["setup"]["scheme"])  # noqa: E731
    if len({key(r) for r in reports}) != 1:
        raise ValueError(f"can only merge one split/scheme, got {sorted({key(r) for r in reports})}")
    for r in reports:
        if any("per_seed" not in m for m in r["results"]):
            raise ValueError("report predates per-seed records (v0.2); rerun it to merge")
    per_model: dict[str, list[dict]] = {}
    for r in reports:
        for m in r["results"]:
            per_model.setdefault(m["model"], []).extend(m["per_seed"])
    setup = dict(reports[0]["setup"])
    skipped = sorted({s for r in reports for s in r["skipped_models"]} - set(per_model))
    return build_report(setup, per_model, skipped, reference, n_boot)


def restrict(report: dict, dataset: str, reference: str | None = None, n_boot: int = N_BOOT) -> dict:
    """The same report scored on one dataset's test subjects only, e.g. the Walch
    subjects of a model trained and k-folded on Walch + BIDSleep pooled."""
    prefix = f"{dataset}/"
    per_model: dict[str, list[dict]] = {}
    for m in report["results"]:
        for rec in m["per_seed"]:
            keep = [i for i, s in enumerate(rec["subjects"]) if s.startswith(prefix)]
            if not keep:
                raise ValueError(f"{m['model']}: no {dataset} subjects in this report")
            per_model.setdefault(m["model"], []).append({
                **rec, "subjects": [rec["subjects"][i] for i in keep],
                "subject_confusions": [rec["subject_confusions"][i] for i in keep],
                "nights": [row for row in rec["nights"] if row["night"].startswith(prefix)],
            })
    s = report["setup"]
    setup = {**s, "split": f"{s['split']}_{dataset}", "description": f"{s['description']}; scored on {dataset} only"}
    return build_report(setup, per_model, report["skipped_models"], reference or report.get("reference"), n_boot)


def compare_reports(a: dict, model_a: str, b: dict, model_b: str, n_boot: int = N_BOOT,
                    dataset: str | None = None) -> dict:
    """Paired subject bootstrap of a model in one report against a model in another
    (e.g. trained on more nights), over the test subjects both reports share."""
    ra, rb = _result(a, model_a), _result(b, model_b)
    common = sorted(set(_subjects(ra)) & set(_subjects(rb)))
    if dataset:
        common = [s for s in common if s.startswith(f"{dataset}/")]
    if not common:
        raise ValueError("the two reports share no test subjects")
    pick = lambda r: _subject_stack(r)[[_subjects(r).index(s) for s in common]]  # noqa: E731
    sa, sb = pick(ra), pick(rb)
    kappa = lambda cms: float(metrics.batch_scores(cms.sum(0)[None])["kappa"][0])  # noqa: E731
    return {"model": model_a, "reference": model_b, "n_subjects": len(common),
            "kappa": kappa(sa), "reference_kappa": kappa(sb), **metrics.paired_bootstrap(sa, sb, n_boot)}


def _result(report: dict, model: str) -> dict:
    r = next((r for r in report["results"] if r["model"] == model), None)
    if r is None:
        raise ValueError(f"model {model!r} is not in this report")
    return r


# -- output --------------------------------------------------------------------

def write(report: dict, out_dir: Path, tag: str | None = None) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = report["setup"]
    stem = f"{s['split']}-{s['scheme']}" + (f"-{tag}" if tag else "")
    jpath, mpath = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    jpath.write_text(json.dumps(report, indent=1, default=_json_default))
    mpath.write_text(to_markdown(report))
    return jpath, mpath


def to_markdown(report: dict) -> str:
    s = report["setup"]
    classes = s["classes"]
    results = report["results"]
    n_seeds = sorted({len(r["seeds"]) for r in results})
    lines = [
        f"# OpenVitals benchmark: {s['split']} / {s['scheme']}",
        "",
        f"- Data: {s['description']}",
        f"- Folds: {len(s['folds'])} (subject-wise; no subject in both train and test)",
        f"- Seeds per model: {', '.join(map(str, n_seeds))}; CIs: 95% subject bootstrap "
        f"({results[0]['ci']['n_boot'] if results else N_BOOT} draws)",
        f"- openvitals {report['openvitals_version']}, Python {report['python']}",
        "",
        "## Epoch agreement (pooled over test epochs and seeds)",
        "",
        "| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | "
        + " | ".join(f"{c} sens" for c in classes) + " | subject kappa (mean ± sd) |",
        "|---|---|---|---|---|---|" + "---|" * len(classes) + "---|",
    ]
    for r in results:
        p, ci, ps = r["pooled"], r["ci"], r["per_subject"]
        sens = " | ".join(_pct(p["per_class"][c]["sensitivity"]) for c in classes)
        sd = f"{r['seed_kappa']['sd']:.3f}" if len(r["seeds"]) > 1 else "–"
        lines.append(
            f"| {r['model']} | {len(r['seeds'])} | {p['kappa']:.3f} [{ci['kappa'][0]:.3f}, {ci['kappa'][1]:.3f}] | "
            f"{sd} | {_pct(p['accuracy'])} [{_pct(ci['accuracy'][0])}, {_pct(ci['accuracy'][1])}] | "
            f"{p['macro_f1']:.3f} | {sens} | {ps['kappa_mean']:.3f} ± {ps['kappa_sd']:.3f} |"
        )
    if report.get("comparisons"):
        lines += [
            "",
            f"## Paired comparison against `{report['reference']}` (same subject draws)",
            "",
            "| model | Δ kappa | 95% CI | P(Δ ≤ 0) |",
            "|---|---|---|---|",
        ]
        for c in sorted(report["comparisons"], key=lambda c: -c["delta_kappa"]):
            lines.append(f"| {c['model']} | {c['delta_kappa']:+.3f} | [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | "
                         f"{c['p_le_0']:.3f} |")
    lines += [
        "",
        "## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)",
        "",
        "| model | TST min | WASO min | sleep efficiency | SOL min |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        a = r["night_agreement"]
        lines.append(f"| {r['model']} | " + " | ".join(_ba(a[k], pct=(k == "se")) for k in SUMMARY_FIELDS) + " |")
    if report["skipped_models"]:
        lines += ["", f"Skipped (scheme not supported): {', '.join(report['skipped_models'])}"]
    lines.append("")
    return "\n".join(lines)


def _ba(b: dict, pct: bool = False) -> str:
    if not b["n"]:
        return "–"
    f = (lambda x: f"{100 * x:+.1f}%") if pct else (lambda x: f"{x:+.1f}")
    return f"{f(b['bias'])} [{f(b['loa_low'])}, {f(b['loa_high'])}]"


def _pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def _nanmean(a) -> float:
    a = np.asarray(a, float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _nanstd(a) -> float:
    a = np.asarray(a, float)
    return float(np.nanstd(a, ddof=1)) if np.isfinite(a).sum() > 1 else float("nan")


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
