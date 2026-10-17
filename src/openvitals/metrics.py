"""Epoch-level agreement and night-level sleep summary metrics.

Epoch metrics are reported two ways:
- pooled: all test epochs concatenated (what most papers report)
- per-subject mean: each subject weighted equally (harder to game with one long night)
"""

from __future__ import annotations

import numpy as np

from .schema import EPOCH_SECONDS


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(m, (y_true, y_pred), 1)
    return m


def cohen_kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def epoch_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    k = len(class_names)
    cm = confusion(y_true, y_pred, k)
    support = cm.sum(1)
    predicted = cm.sum(0)
    tp = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.where(support > 0, tp / support, np.nan)
        prec = np.where(predicted > 0, tp / predicted, np.nan)
        f1 = np.where(sens + prec > 0, 2 * sens * prec / (sens + prec), 0.0)
    f1 = np.where(support > 0, f1, np.nan)
    n = cm.sum()
    return {
        "n_epochs": int(n),
        "accuracy": float(tp.sum() / n) if n else float("nan"),
        "kappa": cohen_kappa(cm),
        "macro_f1": float(np.nanmean(f1)) if np.any(support > 0) else float("nan"),
        "per_class": {
            c: {"support": int(support[i]), "sensitivity": _f(sens[i]), "precision": _f(prec[i]), "f1": _f(f1[i])}
            for i, c in enumerate(class_names)
        },
        "confusion": cm.tolist(),
    }


def sleep_summary(y: np.ndarray, sleep_classes: set[int]) -> dict:
    """Night summary from a hypnogram in scheme indices (valid epochs only, in order).

    TST: total sleep time; SE: sleep efficiency; SOL: sleep onset latency;
    WASO: wake after sleep onset. Minutes unless noted.
    """
    asleep = np.isin(y, list(sleep_classes))
    epoch_min = EPOCH_SECONDS / 60
    tib = len(y) * epoch_min
    tst = float(asleep.sum() * epoch_min)
    if asleep.any():
        onset = int(np.argmax(asleep))
        last = len(asleep) - 1 - int(np.argmax(asleep[::-1]))
        sol = onset * epoch_min
        waso = float((~asleep[onset:last + 1]).sum() * epoch_min)
    else:
        sol, waso = tib, 0.0
    return {"tib_min": tib, "tst_min": tst, "se": tst / tib if tib else float("nan"),
            "sol_min": sol, "waso_min": waso}


def bland_altman(pred: np.ndarray, true: np.ndarray) -> dict:
    """Bias and 95% limits of agreement for a per-night quantity."""
    d = np.asarray(pred, float) - np.asarray(true, float)
    if d.size == 0:
        return {"n": 0, "bias": float("nan"), "loa_low": float("nan"), "loa_high": float("nan"), "mae": float("nan")}
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    bias = float(d.mean())
    return {"n": int(d.size), "bias": bias, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
            "mae": float(np.abs(d).mean())}


def _f(x) -> float | None:
    return None if np.isnan(x) else float(x)


# -- confusion-matrix metrics and subject bootstrap ----------------------------
#
# Subjects, not epochs or nights, are the independent unit: epochs within a night
# are strongly autocorrelated, and nights of one person are not independent.
# So confidence intervals resample subjects with replacement, summing each
# draw's per-subject confusion matrices. Paired comparisons reuse the same
# draws for both models, which removes between-subject variance from the difference.


def metrics_from_confusion(cm: np.ndarray, class_names: list[str]) -> dict:
    """Same output as `epoch_metrics`, from a confusion matrix."""
    cm = np.asarray(cm, dtype=np.int64)
    k = len(class_names)
    y_true = np.repeat(np.repeat(np.arange(k), k), cm.ravel())
    y_pred = np.repeat(np.tile(np.arange(k), k), cm.ravel())
    return epoch_metrics(y_true, y_pred, class_names)


def batch_scores(cms: np.ndarray) -> dict[str, np.ndarray]:
    """Kappa, accuracy, macro F1 and per-class sensitivity for a stack (B, k, k)."""
    cms = np.asarray(cms, dtype=np.float64)
    n = cms.sum(axis=(1, 2))
    tp = np.diagonal(cms, axis1=1, axis2=2)
    rows, cols = cms.sum(2), cms.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        po = tp.sum(1) / n
        pe = (rows * cols).sum(1) / n**2
        kappa = (po - pe) / (1 - pe)
        sens = np.where(rows > 0, tp / rows, np.nan)
        prec = np.where(cols > 0, tp / cols, np.nan)
        f1 = np.where(sens + prec > 0, 2 * sens * prec / (sens + prec), 0.0)
        f1 = np.where(rows > 0, f1, np.nan)
        macro_f1 = np.nanmean(f1, axis=1)
    return {"kappa": kappa, "accuracy": po, "macro_f1": macro_f1, "sensitivity": sens}


def subject_bootstrap(subject_cms: np.ndarray, class_names: list[str], n_boot: int = 2000,
                      seed: int = 0, level: float = 0.95) -> dict:
    """Percentile CIs from resampling subjects. subject_cms: (S, k, k)."""
    idx = _draws(len(subject_cms), n_boot, seed)
    s = batch_scores(np.asarray(subject_cms)[idx].sum(1))
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    ci = lambda a: [float(np.nanpercentile(a, lo)), float(np.nanpercentile(a, hi))]  # noqa: E731
    return {
        "n_boot": n_boot, "level": level,
        "kappa": ci(s["kappa"]), "accuracy": ci(s["accuracy"]), "macro_f1": ci(s["macro_f1"]),
        "sensitivity": {c: ci(s["sensitivity"][:, i]) for i, c in enumerate(class_names)},
    }


def paired_bootstrap(cms_a: np.ndarray, cms_b: np.ndarray, n_boot: int = 2000, seed: int = 0,
                     level: float = 0.95) -> dict:
    """Kappa(a) − kappa(b) over the same subject draws. Both (S, k, k), same subject order.
    p_le_0 is the bootstrap probability that a is not better than b."""
    a, b = np.asarray(cms_a), np.asarray(cms_b)
    if a.shape != b.shape:
        raise ValueError(f"subject sets differ: {a.shape} vs {b.shape}")
    idx = _draws(len(a), n_boot, seed)
    d = batch_scores(a[idx].sum(1))["kappa"] - batch_scores(b[idx].sum(1))["kappa"]
    point = float(batch_scores(a.sum(0)[None])["kappa"][0] - batch_scores(b.sum(0)[None])["kappa"][0])
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return {"delta_kappa": point, "ci": [float(np.nanpercentile(d, lo)), float(np.nanpercentile(d, hi))],
            "p_le_0": float(np.mean(d <= 0)), "n_boot": n_boot}


def _draws(n_subjects: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_subjects, size=(n_boot, n_subjects))
