# Model card: `stager_hr_4class` (OpenVitals 0.6.0)

**Not a medical device.** Sleep-stage estimates for wellness, research and open-source
wearable apps. Don't use them to diagnose or treat anything.

## What it does

It classifies each 30-second epoch of a night as **wake, light (N1/N2), deep (N3) or REM**,
from **wrist heart rate only**. Motion is not used (see "Why heart rate only").

- **Input:** heart-rate samples (time, bpm) for one night. The 18 per-epoch features come from
  `openvitals.features.extract(night, "hr")`.
- **Output:** per-epoch class probabilities. `openvitals stage` / `Stager` add the argmax stage,
  night totals (total sleep time, efficiency, onset latency, wake after sleep onset, minutes per
  stage) and data-quality warnings.
- **Architecture:** an ensemble of 10 small whole-night sequence models: 5 dilated TCNs and 5
  bidirectional GRUs (seeds 0–4), averaging their softmax. They form one ONNX graph (opset 17) with
  feature standardisation inside. The file is `src/openvitals/assets/stager_hr_4class.onnx`.
- **Runtime:** numpy + onnxruntime on CPU. A night stages in well under a second.

## Training data

| dataset | subjects / nights | device | labels |
|---|---|---|---|
| BIDSleep (PhysioNet 1.0.1) | 47 / 253 | Apple Watch, HealthKit HR (~0.2 Hz), home | Dreem 2 EEG headband, expert-corrected |
| Walch (PhysioNet sleep-accel 1.0.0) | 31 / 31 | Apple Watch PPG HR, lab | polysomnography |

The release model is trained on **all 284 nights**, so it has no held-out score of its own. Its
expected accuracy comes from the same recipe trained without Walch and tested on Walch (below).
Adults only; small cohorts; both datasets use Apple Watch heart rate.

## Expected accuracy

**Same recipe, trained on BIDSleep only (253 nights), tested on Walch (31 subjects, lab PSG).**
Five seeds pooled within the ensemble; 95% CIs from resampling subjects (benchmark v0.5).

| metric | value |
|---|---|
| Cohen's kappa (4-class) | **0.514** [0.469, 0.558] |
| accuracy | 66.6% [63.6%, 69.5%] |
| sensitivity: wake / light / deep / REM | 54% / 56% / 83% / 88% |
| total sleep time, bias [95% limits of agreement] | −0.3 min [−38, +38] |
| wake after sleep onset, bias [LoA] | −3.6 min [−37, +29] |

For context, one Oura-funded study reported 4-class kappa of 0.65 (Oura), 0.60 (Apple Watch) and
0.55 (Fitbit) on their own cohort. Cohorts differ, so these numbers aren't directly comparable.

### Release-path check and robustness to coarser heart rate

`scripts/validate_release.py` exports the BIDSleep-only recipe to ONNX and runs it the way users do.
The first row (stored features, benchmark mask) reproduces the benchmark exactly, which shows
the export is lossless. The raw-HR rows go from raw samples through `night_from_samples` →
`Stager`. The thinned rows keep one heart-rate sample per 30 s, 60 s or 5 min, as many devices
log at night.

| path | kappa [95% CI] | accuracy | wake sens | light sens | deep sens | rem sens | epochs |
|---|---|---|---|---|---|---|---|
| stored features, benchmark mask | 0.514 [0.469, 0.558] | 66.6% | 54% | 56% | 83% | 88% | 25962 |
| raw HR, native (~5 s) | 0.513 [0.467, 0.557] | 66.5% | 54% | 56% | 83% | 88% | 26025 |
| raw HR, 1 sample / 30 s | 0.476 [0.434, 0.520] | 63.8% | 53% | 53% | 75% | 88% | 26302 |
| raw HR, 1 sample / 60 s | 0.455 [0.408, 0.501] | 61.8% | 54% | 51% | 80% | 80% | 26691 |
| raw HR, 1 sample / 300 s | 0.306 [0.261, 0.350] | 50.2% | 37% | 41% | 83% | 58% | 24643 |

At one sample per minute, typical of fitness bands exported via Gadgetbridge, expect kappa around
**0.45**. Every 5 minutes is too coarse to be useful, and `stage` warns about it.

## Known limitations

- **Wake is the weakest class** (about 54% sensitivity). Light sleep is often confused with wake
  and deep. Treat single-night minute counts as rough; trends over weeks are more reliable.
- **Night totals are noisy per night:** total-sleep-time limits of agreement are about ±40 min.
- **Coarser heart rate lowers accuracy** (see robustness above). Many devices log heart rate once a
  minute at night; `stage` warns when sampling is much coarser than the ~5 s training data.
- **Cohort and device:** both training sets are Apple Watch heart rate in adults. Other optical
  sensors, children, and people with arrhythmias, beta-blockers or sleep disorders are untested.
- **Night windowing:** `stage` finds the main sleep period with a first pass and re-stages around
  it. Naps and split sleep are not handled. Use `--whole` or `--no-trim` to control the window.

## Why heart rate only

In v0.4 (motion and heart rate on both datasets), motion added no resolvable kappa to the
sequence models, and it made lab-to-home transfer worse (−0.020 [−0.041, −0.000] kappa,
Walch → BIDSleep). More heart-rate training nights helped far more (+0.05–0.07). See
`RESULTS.md`, v0.4 and v0.5.

## Reproduce

```bash
uv sync --extra seq
uv run openvitals export bidsleep=features:feat/hr253 walch=features:feat \
    --model ens10_hr --scheme 4class --jobs 8 --out src/openvitals/assets/stager_hr_4class.onnx
uv run python scripts/validate_release.py build/val_ens10_bidsleep253.onnx   # held-out check
```

Data licences: Walch is ODC-By 1.0; BIDSleep is per its PhysioNet page. This repository and the
model file contain no raw data.
