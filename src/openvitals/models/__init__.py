"""Models. Each has `name`, `schemes` (supported label schemes),
`fit(nights, scheme)` and `predict(night) -> scheme indices per epoch`.

Sequence models need the `seq` extra (torch); they are imported lazily.
"""

from __future__ import annotations

from .baselines import ColeKripkeStyle, FeatureModel, Majority


def _seq(arch: str, inputs: str):
    def make(seed: int = 0):
        from .sequence import SequenceModel

        return SequenceModel(arch, inputs, seed=seed)
    return make


BASELINES = {
    "majority": lambda seed=0: Majority(),  # deterministic: seed unused
    "cole_kripke": lambda seed=0: ColeKripkeStyle(),
    "logreg_motion": lambda seed=0: FeatureModel("logreg", "motion", seed=seed),
    "logreg_hr": lambda seed=0: FeatureModel("logreg", "hr", seed=seed),
    "logreg_motion_hr": lambda seed=0: FeatureModel("logreg", "motion+hr", seed=seed),
    "gbdt_motion": lambda seed=0: FeatureModel("gbdt", "motion", seed=seed),
    "gbdt_hr": lambda seed=0: FeatureModel("gbdt", "hr", seed=seed),
    "gbdt_motion_hr": lambda seed=0: FeatureModel("gbdt", "motion+hr", seed=seed),
}
SEQUENCE = {f"{a}_{i.replace('+', '_')}": _seq(a, i) for a in ("tcn", "gru") for i in ("hr", "motion+hr")}
REGISTRY = {**BASELINES, **SEQUENCE}


def make(name: str, seed: int = 0):
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](seed=seed)
