from __future__ import annotations

import numpy as np

from .. import features, stages
from ..schema import Night


def training_matrix(nights: list[Night], scheme: str, inputs: str = "motion+hr"):
    """Stack features and scheme labels over nights, keeping labelled, scorable epochs."""
    Xs, ys = [], []
    for n in nights:
        f = features.extract(n, inputs)
        y = stages.to_scheme(n.labels, scheme)
        m = (y >= 0) & f.scorable
        Xs.append(f.X[m])
        ys.append(y[m])
    if not Xs:
        raise ValueError("no training nights")
    return np.vstack(Xs), np.concatenate(ys)


class Majority:
    """Predicts the most common training class everywhere. The floor every model must beat."""

    name = "majority"
    schemes = tuple(stages.SCHEMES)

    def fit(self, nights: list[Night], scheme: str) -> "Majority":
        y = np.concatenate([stages.to_scheme(n.labels, scheme) for n in nights])
        y = y[y >= 0]
        self.label_ = int(np.bincount(y).argmax())
        return self

    def predict(self, night: Night) -> np.ndarray:
        return np.full(night.n_epochs, self.label_, dtype=np.int8)


class ColeKripkeStyle:
    """Sleep/wake from a weighted moving sum of activity (Cole et al. 1992).

    Adaptation: the original works on per-minute ActiGraph counts with weights
    for offsets -4..+2 minutes. We apply the same weights to 30 s epochs of
    log acceleration-magnitude SD and learn the wake threshold on the training
    folds, maximising balanced accuracy, because it is not on the count scale.
    Magnitude SD, not ENMO: on Walch the resting ENMO floor varies 0-1.1 (log
    units) between subjects with device calibration, which cut kappa from 0.29
    to 0.10. Sleep/wake only.
    """

    name = "cole_kripke"
    schemes = ("2class",)
    WEIGHTS = np.array([106, 54, 58, 76, 230, 74, 67], dtype=float)  # offsets -4..+2
    OFFSETS = np.arange(-4, 3)

    def fit(self, nights: list[Night], scheme: str) -> "ColeKripkeStyle":
        _require(scheme, self.schemes)
        scores, ys = [], []
        for n in nights:
            f = features.extract(n, "motion")
            y = stages.to_scheme(n.labels, scheme)
            m = (y >= 0) & f.motion_cov
            scores.append(self._score(f)[m])
            ys.append(y[m])
        s, y = np.concatenate(scores), np.concatenate(ys)
        candidates = np.quantile(s, np.linspace(0.01, 0.99, 197))
        best, best_bacc = candidates[0], -1.0
        for th in candidates:
            pred = np.where(s >= th, 0, 1)
            bacc = _balanced_accuracy(y, pred, 2)
            if bacc > best_bacc:
                best, best_bacc = th, bacc
        self.threshold_ = float(best)
        return self

    def predict(self, night: Night) -> np.ndarray:
        s = self._score(features.extract(night, "motion"))
        return np.where(s >= self.threshold_, 0, 1).astype(np.int8)

    def _score(self, f: features.EpochFeatures) -> np.ndarray:
        a = f.X[:, f.names.index("mag_std_log")]
        n = len(a)
        padded = np.pad(a, (4, 2), mode="edge")
        return sum(w * padded[4 + o: 4 + o + n] for w, o in zip(self.WEIGHTS, self.OFFSETS)) / self.WEIGHTS.sum()


class FeatureModel:
    """Logistic regression or gradient-boosted trees on `features.extract`.

    Class weights are balanced so rare stages (deep, wake) are not ignored.
    """

    schemes = tuple(stages.SCHEMES)

    def __init__(self, kind: str = "gbdt", inputs: str = "motion+hr", seed: int = 0):
        self.kind, self.inputs, self.seed = kind, inputs, seed
        self.name = f"{kind}_{inputs.replace('+', '_')}"

    def fit(self, nights: list[Night], scheme: str) -> "FeatureModel":
        X, y = training_matrix(nights, scheme, self.inputs)
        self.n_classes_ = len(stages.class_names(scheme))
        self.model_ = self._build()
        weights = _balanced_weights(y, self.n_classes_)
        if self.kind == "logreg":
            self.model_.fit(X, y, logisticregression__sample_weight=weights)
        else:
            self.model_.fit(X, y, sample_weight=weights)
        return self

    def predict(self, night: Night) -> np.ndarray:
        f = features.extract(night, self.inputs)
        if night.n_epochs == 0:
            return np.empty(0, dtype=np.int8)
        return self.model_.predict(f.X).astype(np.int8)

    def _build(self):
        if self.kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        if self.kind == "gbdt":
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=self.seed,
            )
        raise ValueError(f"unknown kind {self.kind!r}")


def _balanced_weights(y: np.ndarray, k: int) -> np.ndarray:
    counts = np.bincount(y, minlength=k).astype(float)
    w = np.where(counts > 0, len(y) / (k * np.maximum(counts, 1)), 0.0)
    return w[y]


def _balanced_accuracy(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    recalls = [np.mean(pred[y == c] == c) for c in range(k) if np.any(y == c)]
    return float(np.mean(recalls)) if recalls else 0.0


def _require(scheme: str, allowed: tuple[str, ...]) -> None:
    if scheme not in allowed:
        raise ValueError(f"scheme {scheme!r} not supported; this model supports {allowed}")
