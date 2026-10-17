"""Models. Each has `name`, `schemes` (supported label schemes),
`fit(nights, scheme)` and `predict(night) -> scheme indices per epoch`.

Sequence models need the `seq` extra (torch); they are imported lazily.
"""

from __future__ import annotations

from .baselines import ColeKripkeStyle, FeatureModel, Majority


def _seq(arch: str, inputs: str):
    def make():
        from .sequence import SequenceModel

        return SequenceModel(arch, inputs)
    return make


BASELINES = {
    "majority": lambda: Majority(),
    "cole_kripke": lambda: ColeKripkeStyle(),
    "logreg_motion": lambda: FeatureModel("logreg", "motion"),
    "logreg_hr": lambda: FeatureModel("logreg", "hr"),
    "logreg_motion_hr": lambda: FeatureModel("logreg", "motion+hr"),
    "gbdt_motion": lambda: FeatureModel("gbdt", "motion"),
    "gbdt_hr": lambda: FeatureModel("gbdt", "hr"),
    "gbdt_motion_hr": lambda: FeatureModel("gbdt", "motion+hr"),
}
SEQUENCE = {f"{a}_{i.replace('+', '_')}": _seq(a, i) for a in ("tcn", "gru") for i in ("hr", "motion+hr")}
REGISTRY = {**BASELINES, **SEQUENCE}


def make(name: str):
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name]()
