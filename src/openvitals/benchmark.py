"""Run models over subject-wise folds and score them."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__, features, metrics, models, stages
from .schema import Night
from .splits import Fold

SUMMARY_FIELDS = ("tst_min", "waso_min", "se", "sol_min")


@dataclass
class ModelResult:
    model: str
    scheme: str
    pooled: dict
    per_subject: dict
    night_agreement: dict
    nights: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def evaluate(model_name: str, folds: list[Fold], scheme: str) -> ModelResult | None:
    """Fit on each fold's train nights, predict its test nights, score everything together.
    Returns None if the model does not support `scheme`."""
    if scheme not in models.make(model_name).schemes:
        return None
    names = stages.class_names(scheme)
    sleep = stages.sleep_index(scheme)
    t0 = time.perf_counter()
    all_true, all_pred = [], []
    by_subject: dict[str, tuple[list, list]] = {}
    night_rows = []
    for fold in folds:
        model = models.make(model_name).fit(fold.train, scheme)
        for night in fold.test:
            y_true, y_pred = _scored_epochs(night, model.predict(night), scheme)
            if y_true.size == 0:
                continue
            all_true.append(y_true)
            all_pred.append(y_pred)
            subj = by_subject.setdefault(f"{night.dataset}/{night.subject_id}", ([], []))
            subj[0].append(y_true)
            subj[1].append(y_pred)
            st, sp = metrics.sleep_summary(y_true, sleep), metrics.sleep_summary(y_pred, sleep)
            night_rows.append({
                "night": night.key, "fold": fold.name, "n_epochs": int(y_true.size),
                "kappa": metrics.cohen_kappa(metrics.confusion(y_true, y_pred, len(names))),
                **{f"true_{k}": st[k] for k in SUMMARY_FIELDS},
                **{f"pred_{k}": sp[k] for k in SUMMARY_FIELDS},
            })
    if not all_true:
        raise ValueError("no scorable test epochs")
    pooled = metrics.epoch_metrics(np.concatenate(all_true), np.concatenate(all_pred), names)

    subj_kappa, subj_acc = [], []
    for yt, yp in by_subject.values():
        m = metrics.epoch_metrics(np.concatenate(yt), np.concatenate(yp), names)
        subj_kappa.append(m["kappa"])
        subj_acc.append(m["accuracy"])
    per_subject = {
        "n_subjects": len(by_subject),
        "kappa_mean": _nanmean(subj_kappa), "kappa_sd": _nanstd(subj_kappa),
        "accuracy_mean": _nanmean(subj_acc), "accuracy_sd": _nanstd(subj_acc),
    }
    agreement = {
        k: metrics.bland_altman([r[f"pred_{k}"] for r in night_rows], [r[f"true_{k}"] for r in night_rows])
        for k in SUMMARY_FIELDS
    }
    return ModelResult(model_name, scheme, pooled, per_subject, agreement, night_rows,
                       time.perf_counter() - t0)


def _scored_epochs(night: Night, pred: np.ndarray, scheme: str) -> tuple[np.ndarray, np.ndarray]:
    """Epochs that are both labelled and have sensor data. Same mask for every model,
    so results are comparable."""
    y = stages.to_scheme(night.labels, scheme)
    mask = (y >= 0) & features.extract(night, "motion").scorable
    return y[mask].astype(np.int64), np.asarray(pred)[mask].astype(np.int64)


def run(model_names: list[str], folds: list[Fold], scheme: str, setup: dict) -> dict:
    results = [r for name in model_names if (r := evaluate(name, folds, scheme)) is not None]
    skipped = [n for n in model_names if n not in {r.model for r in results}]
    return {
        "openvitals_version": __version__,
        "python": platform.python_version(),
        "setup": {**setup, "scheme": scheme, "classes": stages.class_names(scheme),
                  "folds": [{"name": f.name, "n_train": len(f.train), "n_test": len(f.test)} for f in folds]},
        "skipped_models": skipped,
        "results": [r.__dict__ for r in results],
    }


def write(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = report["setup"]
    stem = f"{s['split']}-{s['scheme']}"
    jpath, mpath = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    jpath.write_text(json.dumps(report, indent=2, default=_json_default))
    mpath.write_text(to_markdown(report))
    return jpath, mpath


def to_markdown(report: dict) -> str:
    s = report["setup"]
    classes = s["classes"]
    lines = [
        f"# OpenVitals benchmark: {s['split']} / {s['scheme']}",
        "",
        f"- Data: {s['description']}",
        f"- Folds: {len(s['folds'])} (subject-wise; no subject in both train and test)",
        f"- openvitals {report['openvitals_version']}, Python {report['python']}",
        "",
        "## Epoch agreement (pooled over all test epochs)",
        "",
        "| model | accuracy | kappa | macro F1 | " + " | ".join(f"{c} sens" for c in classes) + " | subject kappa (mean ± sd) |",
        "|---|---|---|---|" + "---|" * len(classes) + "---|",
    ]
    for r in report["results"]:
        p, ps = r["pooled"], r["per_subject"]
        sens = " | ".join(_pct(p["per_class"][c]["sensitivity"]) for c in classes)
        lines.append(
            f"| {r['model']} | {_pct(p['accuracy'])} | {p['kappa']:.3f} | {p['macro_f1']:.3f} | {sens} | "
            f"{ps['kappa_mean']:.3f} ± {ps['kappa_sd']:.3f} |"
        )
    lines += [
        "",
        "## Night-level agreement (predicted − PSG; bias [95% limits of agreement])",
        "",
        "| model | TST min | WASO min | sleep efficiency | SOL min |",
        "|---|---|---|---|---|",
    ]
    for r in report["results"]:
        a = r["night_agreement"]
        cells = [_ba(a[k], pct=(k == "se")) for k in SUMMARY_FIELDS]
        lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")
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
    raise TypeError(type(o))
