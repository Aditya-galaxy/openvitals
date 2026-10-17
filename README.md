# OpenVitals

Open, validated sleep and recovery algorithms for wrist wearables, and the
reproducible benchmark that keeps them honest.

Open-source companion apps for Whoop, Oura, Garmin and others can read raw
sensor data, but each one reinvents sleep staging and recovery scoring with no
published accuracy. OpenVitals is a shared, device-agnostic library with
numbers attached.

**Status:** v0.1 – benchmark harness and baselines; see [RESULTS.md](RESULTS.md). Not a medical device; for
wellness and research use only.

## Benchmark rules

- **Subject-wise splits only.** A person's nights never appear in both train and test.
- **Same epochs for every model.** Scored epochs are labelled and have sensor data.
- **Per-class results, not just accuracy.** Wake and deep sleep are where wearables fail.
- **Night-level agreement.** TST, WASO, sleep efficiency and onset latency as
  Bland–Altman bias with 95% limits of agreement.
- **Cross-dataset results are the headline.** Train on one cohort, test on another.

## Datasets

| name | source | subjects / nights | signals | labels | access |
|---|---|---|---|---|---|
| `walch` | [PhysioNet sleep-accel 1.0.0](https://physionet.org/content/sleep-accel/1.0.0/) | 31 / 31 | Apple Watch accel + PPG HR | PSG | open (2.2 GB; fetched from the AWS Open Data mirror) |
| `bidsleep` | [PhysioNet bidsleep-dataset 1.0.1](https://physionet.org/content/bidsleep-dataset/) | 47 / 253 | Apple Watch accel + HealthKit HR | Dreem 2 EEG, expert-corrected | open (27.9 GB, ~27 GB of it motion; loads HR-only if `motion.csv` is absent) |

Planned: DREAMT (credentialed), MESA/SHHS via NSRR (data request). Model
weights trained on restricted data will not be published unless the data-use
agreement allows it.

## Usage

```bash
uv sync
uv run openvitals download walch --dest data          # verified against SHA256SUMS.txt
uv run openvitals download bidsleep --dest data --include '*/hr.csv,*/labels.mat'   # HR-only, ~30 MB
uv run openvitals info walch=data/walch
uv run openvitals bench walch=data/walch --scheme 4class --split loso
uv run openvitals bench bidsleep=data/bidsleep --test walch=data/walch --split cross --scheme 4class
```

Reports go to `results/<split>-<scheme>.{md,json}`; parsed nights are cached in `.cache/`.

Schemes: `2class` (wake/sleep), `3class` (wake/NREM/REM), `4class` (wake/light/deep/REM).

## Baselines

| model | inputs | notes |
|---|---|---|
| `majority` | – | floor every model must beat |
| `cole_kripke` | motion | Cole et al. 1992 weights on 30 s magnitude SD, threshold learned per fold; sleep/wake only |
| `logreg_motion`, `logreg_hr`, `logreg_motion_hr` | features | balanced logistic regression |
| `gbdt_motion`, `gbdt_hr`, `gbdt_motion_hr` | features | balanced histogram gradient boosting |

Features (`openvitals.features`) are numpy-only and computed per night so they
can be ported on-device: ENMO, magnitude SD, z-angle change, centred rolling
windows, per-night z-scored HR and its rolling SD, hours since start.

## Layout

```
src/openvitals/
  schema.py      Night: motion, hr, per-epoch labels, seconds from epoch 0
  stages.py      canonical codes + 2/3/4-class schemes
  datasets/      walch, bidsleep loaders (+ npz cache)
  features.py    per-epoch features
  models/        baselines
  splits.py      subject k-fold, LOSO, cross-dataset
  metrics.py     kappa, per-class, sleep summary, Bland–Altman
  benchmark.py   run + markdown/json report
  download.py    PhysioNet fetch with checksum verification
```

## Tests

```bash
uv run pytest
```

Tests use synthetic nights written in each dataset's on-disk format.
