"""Stage real-world nights with an exported ONNX model. Needs numpy and onnxruntime only.

    from openvitals.infer import Stager, night_from_samples
    night = night_from_samples(hr_unix_seconds, hr_bpm)        # optionally motion=(t, x, y, z)
    result = Stager().stage(night)                             # bundled HR-only 4-class model
    print(result.summary, result.hypnogram())

Models were trained with epoch 0 near lights-off. A window that starts hours before
bed (e.g. a 20:00-12:00 export) shifts the time-of-night feature, so by default the
night is staged twice: once to find the main sleep period, then again on a window
from TRIM_PAD before sleep onset to TRIM_PAD after final wake.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import numpy as np

from . import features, metrics, stages
from .schema import EPOCH_SECONDS, Night

BUNDLED_MODEL = "stager_hr_4class.onnx"
TRIM_PAD = 20 * 60             # seconds kept before sleep onset / after final wake
MAX_WAKE_GAP = 30 * 60         # wake runs shorter than this stay inside the main sleep period
TRAINING_HR_INTERVAL = 5.0     # seconds; Walch and BIDSleep heart rate is ~0.2 Hz


def default_model_path() -> Path:
    return Path(str(resources.files("openvitals") / "assets" / BUNDLED_MODEL))


def night_from_samples(hr_t, hr_bpm, motion=None, start: float | None = None, end: float | None = None,
                       night_id: str = "night", subject_id: str = "user", dataset: str = "user") -> Night:
    """Build an unlabelled Night from raw samples with absolute Unix times (seconds).

    motion: optional (n, 4) array of unix_t, x, y, z in g. start/end default to the
    span of the heart-rate samples. Epoch 0 starts at `start`."""
    hr_t = np.asarray(hr_t, dtype=float)
    hr_bpm = np.asarray(hr_bpm, dtype=float)
    ok = np.isfinite(hr_t) & np.isfinite(hr_bpm) & (hr_bpm > 25) & (hr_bpm < 230)
    hr_t, hr_bpm = hr_t[ok], hr_bpm[ok]
    if hr_t.size == 0:
        raise ValueError("no valid heart-rate samples")
    start = float(np.floor(hr_t.min())) if start is None else float(start)
    end = float(hr_t.max()) if end is None else float(end)
    n_epochs = int(np.ceil((end - start) / EPOCH_SECONDS))
    if n_epochs <= 0:
        raise ValueError("empty time window")
    keep = (hr_t >= start) & (hr_t < start + n_epochs * EPOCH_SECONDS)
    hr = np.column_stack([hr_t[keep] - start, hr_bpm[keep]])
    mot = np.empty((0, 4))
    if motion is not None and len(motion):
        mot = np.asarray(motion, dtype=float).copy()
        mot[:, 0] -= start
    return Night(dataset, subject_id, night_id, mot, hr, np.full(n_epochs, stages.UNKNOWN, dtype=np.int8),
                 meta={"unix_start": start})


@dataclass
class StagingResult:
    unix_start: float                 # time of epoch 0
    classes: list[str]
    stage: np.ndarray                 # class index per 30 s epoch, -1 = no sensor data
    probs: np.ndarray                 # (epochs, classes)
    summary: dict
    quality: dict
    warnings: list[str] = field(default_factory=list)

    def times(self) -> np.ndarray:
        return self.unix_start + EPOCH_SECONDS * np.arange(len(self.stage))

    def stage_names(self) -> list[str]:
        return ["no_data" if s < 0 else self.classes[s] for s in self.stage]

    def to_csv(self, path: Path, tz=None) -> Path:
        tz = tz or timezone.utc
        lines = ["time,stage," + ",".join(f"p_{c}" for c in self.classes)]
        for t, name, p in zip(self.times(), self.stage_names(), self.probs):
            iso = datetime.fromtimestamp(t, tz).isoformat(timespec="seconds")
            lines.append(f"{iso},{name}," + ",".join(f"{v:.4f}" for v in p))
        Path(path).write_text("\n".join(lines) + "\n")
        return Path(path)

    def to_dict(self, tz=None) -> dict:
        tz = tz or timezone.utc
        iso = lambda t: datetime.fromtimestamp(t, tz).isoformat(timespec="seconds")  # noqa: E731
        return {"start": iso(self.unix_start), "end": iso(self.unix_start + EPOCH_SECONDS * len(self.stage)),
                "epoch_seconds": EPOCH_SECONDS, "classes": self.classes, "summary": self.summary,
                "quality": self.quality, "warnings": self.warnings}

    def hypnogram(self, minutes_per_char: int = 5) -> str:
        """Text hypnogram, one character per `minutes_per_char` (majority stage).
        Rows top to bottom: wake, REM, then lighter to deeper sleep; '·' = no data."""
        k = max(1, minutes_per_char * 60 // EPOCH_SECONDS)
        major = []
        for i in range(0, len(self.stage), k):
            b = self.stage[i:i + k]
            b = b[b >= 0]
            major.append(int(np.bincount(b, minlength=len(self.classes)).argmax()) if b.size else -1)
        order = sorted(range(len(self.classes)), key=lambda i: _DISPLAY.index(self.classes[i])
                       if self.classes[i] in _DISPLAY else len(_DISPLAY) + i)
        width = max(len(c) for c in self.classes)
        return "\n".join(f"{self.classes[ci]:>{width}} |" + "".join("█" if m == ci else "·" if m < 0 else " "
                                                                   for m in major) for ci in order)


_DISPLAY = ["wake", "rem", "sleep", "nrem", "light", "deep"]


class Stager:
    """Load an exported OpenVitals ONNX model (single model or ensemble) and stage nights."""

    def __init__(self, model_path: Path | None = None):
        import onnxruntime as ort

        self.path = Path(model_path) if model_path else default_model_path()
        if not self.path.exists():
            raise FileNotFoundError(f"model not found: {self.path}")
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.meta = dict(meta)
        self.classes = json.loads(meta["classes"])
        self.feature_names = json.loads(meta["features"])
        self.inputs = meta["inputs"]
        self.scheme = meta.get("scheme", "")
        self.output = self.session.get_outputs()[0].name   # "probs" (ensemble) or "logits"
        self.sleep_classes = set(range(1, len(self.classes)))

    def predict_proba(self, night: Night) -> tuple[np.ndarray, np.ndarray]:
        f = features.extract(night, self.inputs)
        if f.names != self.feature_names:
            raise ValueError(f"feature mismatch: model wants {self.feature_names}, got {f.names}")
        out = self.session.run([self.output], {"features": f.X.astype(np.float32)[None]})[0][0]
        if self.output != "probs":
            e = np.exp(out - out.max(1, keepdims=True))
            out = e / e.sum(1, keepdims=True)
        has_data = _near_hr(night) if self.inputs == "hr" else f.scorable
        return out, has_data

    def stage(self, night: Night, trim: bool = True) -> StagingResult:
        start = night.meta.get("unix_start", 0.0)
        probs, has_data = self.predict_proba(night)
        if trim:
            window = _main_sleep_window(probs.argmax(1), has_data)
            if window is not None:
                lo = max(0, window[0] - TRIM_PAD // EPOCH_SECONDS)
                hi = min(len(probs), window[1] + 1 + TRIM_PAD // EPOCH_SECONDS)
                if (lo, hi) != (0, len(probs)):
                    night = _slice(night, lo, hi)
                    start += lo * EPOCH_SECONDS
                    probs, has_data = self.predict_proba(night)
        stage = probs.argmax(1).astype(np.int8)
        stage[~has_data] = stages.UNKNOWN
        scored = stage[stage >= 0]
        summary = metrics.sleep_summary(scored, self.sleep_classes) if scored.size else {}
        summary.update({f"{c}_min": float((scored == i).sum() * EPOCH_SECONDS / 60) for i, c in enumerate(self.classes)})
        quality, warnings = _quality(night, has_data)
        return StagingResult(start, self.classes, stage, probs, summary, quality, warnings)


MAX_COVER_GAP = 10 * 60  # seconds; an epoch farther than this from any HR sample has no data


def _near_hr(night: Night) -> np.ndarray:
    """Epochs whose centre is near a heart-rate sample. Features interpolate HR across
    gaps, so an epoch between two once-a-minute samples is covered; a gap counts as
    missing data only beyond 2.5x the typical sampling interval (at most MAX_COVER_GAP)."""
    t = night.hr[:, 0]
    if len(t) == 0:
        return np.zeros(night.n_epochs, dtype=bool)
    step = float(np.median(np.diff(t))) if len(t) > 1 else EPOCH_SECONDS
    tol = min(MAX_COVER_GAP, max(EPOCH_SECONDS, 2.5 * step))
    centres = (np.arange(night.n_epochs) + 0.5) * EPOCH_SECONDS
    i = np.clip(np.searchsorted(t, centres), 1, max(len(t) - 1, 1))
    nearest = np.minimum(np.abs(centres - t[i - 1]), np.abs(t[np.minimum(i, len(t) - 1)] - centres))
    return nearest <= tol


def _main_sleep_window(pred: np.ndarray, has_data: np.ndarray) -> tuple[int, int] | None:
    """Longest run of sleep epochs, bridging wake (or no-data) gaps shorter than MAX_WAKE_GAP."""
    asleep = (pred > 0) & has_data
    idx = np.flatnonzero(asleep)
    if idx.size == 0:
        return None
    max_gap = MAX_WAKE_GAP // EPOCH_SECONDS
    best, cur_start, prev = (idx[0], idx[0]), idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap:
            if prev - cur_start > best[1] - best[0]:
                best = (cur_start, prev)
            cur_start = i
        prev = i
    if prev - cur_start > best[1] - best[0]:
        best = (cur_start, prev)
    return int(best[0]), int(best[1])


def _slice(night: Night, lo: int, hi: int) -> Night:
    t0, t1 = lo * EPOCH_SECONDS, hi * EPOCH_SECONDS
    def cut(a):
        if not len(a):
            return a
        a = a[(a[:, 0] >= t0) & (a[:, 0] < t1)].copy()
        a[:, 0] -= t0
        return a
    return Night(night.dataset, night.subject_id, night.night_id, cut(night.motion), cut(night.hr),
                 night.labels[lo:hi].copy(), meta={**night.meta, "unix_start": night.meta.get("unix_start", 0) + t0})


def _quality(night: Night, has_data: np.ndarray) -> tuple[dict, list[str]]:
    t = night.hr[:, 0]
    interval = float(np.median(np.diff(t))) if len(t) > 1 else float("nan")
    q = {"hours": night.n_epochs * EPOCH_SECONDS / 3600, "hr_samples": int(len(t)),
         "median_hr_interval_s": interval, "epochs_with_data": float(has_data.mean()) if len(has_data) else 0.0}
    w = []
    if np.isfinite(interval) and interval > 3 * TRAINING_HR_INTERVAL:
        w.append(f"heart rate is sampled every ~{interval:.0f} s; the model was trained on ~{TRAINING_HR_INTERVAL:.0f} s "
                 "data, so expect lower accuracy (see MODEL_CARD.md, robustness)")
    if q["epochs_with_data"] < 0.8:
        w.append(f"only {q['epochs_with_data']:.0%} of 30 s epochs have heart-rate data")
    if q["hours"] < 3:
        w.append(f"only {q['hours']:.1f} h of data; staging needs most of a night")
    return q, w
