# OpenVitals

Open, validated sleep and recovery algorithms for wrist wearables, and the
reproducible benchmark that keeps them honest.

Open-source companion apps for Whoop, Oura, Garmin and others can read raw
sensor data, but each one reinvents sleep staging and recovery scoring with no
published accuracy. OpenVitals is a shared, device-agnostic library with
numbers attached.

**Status:** v0.5 – baselines, whole-night sequence models and ensembles, 5-seed results with subject-bootstrap CIs, motion + HR on both datasets, pooled training on all 253 BIDSleep HR nights + Walch; see [RESULTS.md](RESULTS.md). Not a medical device; for
wellness and research use only.

## Benchmark rules

- **Subject-wise splits only.** A person's nights never appear in both train and test.
- **Same epochs for every model.** Scored epochs are labelled and have sensor data.
- **Per-class results, not just accuracy.** Wake and deep sleep are where wearables fail.
- **Night-level agreement.** TST, WASO, sleep efficiency and onset latency as
  Bland–Altman bias with 95% limits of agreement.
- **Cross-dataset results are the headline.** Train on one cohort, test on another.
- **Uncertainty is reported, and subjects are the unit.** 95% CIs resample test subjects, not
  epochs or nights. Seed SD shows training instability. Model differences use a paired subject
  bootstrap against a reference model.

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

### Seeds, confidence intervals and merging

```bash
# 5 training seeds, paired comparison against gbdt_hr
uv run openvitals bench walch=data/walch --models gbdt_hr,tcn_hr --seeds 0,1,2,3,4 --reference gbdt_hr
# or one process per seed, merged afterwards (identical result)
uv run openvitals bench walch=data/walch --models gbdt_hr,tcn_hr --seeds 3 --tag s3
uv run openvitals merge results/kfold-4class-s*.json --reference gbdt_hr --out results/merged
```

Each JSON report keeps one confusion matrix per test subject per seed, so merging is exact,
and comparisons against a different reference can be recomputed without retraining.

```bash
# a pooled k-fold (Walch + BIDSleep) scored on its Walch test subjects only
uv run openvitals merge results/pooled/kfold-4class-s*.json --dataset walch --out results/merged
# paired Δ between models in two reports (e.g. trained on more nights), on their shared test subjects
uv run openvitals compare results/new.json:gru_hr results/old.json:gru_hr --dataset walch
```

### Sequence models and ONNX

```bash
uv sync --extra seq                                   # torch, onnx, onnxruntime
uv run openvitals bench walch=data/walch --models tcn_motion_hr,gru_motion_hr --split loso
uv run openvitals export walch=data/walch --model tcn_motion_hr --scheme 4class --out tcn.onnx
```

`ens_hr` / `ens_motion_hr` average the softmax of a TCN and a GRU trained with the same seed
(not ONNX-exportable yet).

The ONNX graph takes raw `features.extract` output, shape `(1, epochs, n_features)`,
with standardisation inside. Feature order and class names are in the model metadata.

### Large datasets: stream into a feature store

```bash
uv run openvitals featurize bidsleep --data-dir data --features feat --download --evict
uv run openvitals bench bidsleep=features:feat --test walch=features:feat --split cross
```

This handles one subject at a time (download, verify, featurize, delete raw) and
resumes after interruption. [`notebooks/full_benchmark.ipynb`](notebooks/full_benchmark.ipynb)
runs this on Colab with the feature store on Google Drive. Use it for full BIDSleep
when PhysioNet is slow from your network.

### Full BIDSleep on Google Cloud

```bash
cloud/gcp/launch.sh        # bucket + e2-standard-8 VM in us-east1, wheel + Walch features copied, deps installed
gcloud compute ssh openvitals-run --zone us-east1-b --command \
  "cd ov && BUCKET=gs://kronagent-openvitals MAX_NIGHTS=1 tmux new -d -s run 'bash run.sh'"
```

`run.sh` featurizes BIDSleep (streamed and resumable), then runs 5-seed benchmarks in parallel and
merges them. It syncs features, results and logs to the bucket every 10 min, then powers the VM off.
Re-running it skips finished subjects and reports. To deploy a new wheel mid-run, copy it to
`~/ov/` on the VM and run `BUCKET=... bash ov/restart.sh`; partial downloads resume. Download
workers share one queue (`featurize --claim`), so none sits idle while subjects remain.

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
  models/sequence.py  whole-night TCN / bi-GRU + ONNX export
  benchmark.py   run + markdown/json report
  download.py    PhysioNet fetch with checksum verification
  featstore.py   per-subject feature files; streaming featurize
```

## Tests

```bash
uv run pytest
```

Tests use synthetic nights written in each dataset's on-disk format.

## License

Code: MIT (see `LICENSE`). The datasets keep their own licences (see the table above); this repository
contains no raw data. Not a medical device.
