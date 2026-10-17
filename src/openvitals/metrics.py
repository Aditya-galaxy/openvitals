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
