"""Baseline models. Each has `name`, `schemes` (supported label schemes),
`fit(nights, scheme)` and `predict(night) -> scheme indices per epoch`."""

from __future__ import annotations

from .baselines import ColeKripkeStyle, FeatureModel, Majority

REGISTRY = {
    "majority": lambda: Majority(),
    "cole_kripke": lambda: ColeKripkeStyle(),
    "logreg_motion": lambda: FeatureModel("logreg", "motion"),
    "logreg_hr": lambda: FeatureModel("logreg", "hr"),
    "logreg_motion_hr": lambda: FeatureModel("logreg", "motion+hr"),
    "gbdt_motion": lambda: FeatureModel("gbdt", "motion"),
    "gbdt_hr": lambda: FeatureModel("gbdt", "hr"),
    "gbdt_motion_hr": lambda: FeatureModel("gbdt", "motion+hr"),
}


def make(name: str):
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name]()
