# OpenVitals

Open, validated sleep and recovery algorithms for wrist wearables, and the
reproducible benchmark that keeps them honest.

Open-source companion apps for Whoop, Oura, Garmin and others can read raw
sensor data, but each one reinvents sleep staging and recovery scoring with no
published accuracy. OpenVitals is a shared, device-agnostic library with
numbers attached.

**Status:** v0.8 – `openvitals stage` for your own nights (Apple Health, Gadgetbridge, CSV) with a
bundled heart-rate-only model; overnight resting heart rate, HRV and a personal-baseline recovery
score; tested on a held-out cohort (DREAMT) and against ECG. See [RESULTS.md](RESULTS.md). Not a
medical device; for wellness and research use only.

## Accuracy at a glance

| what | tested on | result |
|---|---|---|
| Sleep stages (wake / light / deep / REM) | Walch: 31 healthy adults, Apple Watch, lab PSG, never seen in training | kappa **0.51** [0.47, 0.56] |
| same, heart rate once a minute | Walch | kappa 0.46 |
| same, different people and device | DREAMT: 100 sleep-lab patients (most with apnea), Empatica E4 | kappa **0.24** [0.20, 0.27]; wake/sleep 0.35 |
| Resting heart rate | Walch, BIDSleep | staging errors move it < 1 bpm; night-to-night ICC 0.69 |
| HRV (RMSSD) from wrist pulse | DREAMT, 20 people vs ECG | nightly error 16% (median), ranks people well (ρ 0.88), but reads ~28% high |

Kappa measures agreement with expert scoring (1 = perfect, 0 = chance); published studies report
about 0.55–0.65 for commercial watches on their own cohorts. **The model works best on healthy
adults with Apple Watch-quality heart rate. Accuracy roughly halves for people with sleep disorders
or other wrist sensors.** Wrist HRV is fine for "compared with my usual", not as an absolute number.
Details: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), [`docs/RECOVERY.md`](docs/RECOVERY.md).

## Stage your own nights

```bash
pip install "git+https://github.com/Aditya-galaxy/openvitals"   # or a built wheel
openvitals stage --apple-health ~/Downloads/export.zip --out my-sleep
openvitals stage --gadgetbridge Gadgetbridge --night 2026-11-10
openvitals stage --csv hr.csv --whole      # time (ISO 8601 or Unix s), hr
```

It reads heart rate from the export, cuts it into nights (20:00–12:00 local time by default), finds
the main sleep period, and stages each 30 s epoch as wake, light, deep or REM. Then it prints a
hypnogram and night totals. With `--out` it writes `<date>.csv` (per-epoch stage and probabilities),
`<date>.json` and `nights.csv`. The bundled model is the heart-rate-only 10-model TCN + GRU
ensemble, which runs on numpy + onnxruntime (no torch). Accuracy, training data and limits are in
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md). **Not a medical device.**

Each night also gets **resting heart rate** (lowest 5 minutes asleep) and **average sleeping heart
rate**. With beat-to-beat data, it gets **HRV (RMSSD)** too: pass a chest-strap export with
`--ibi beats.csv` (e.g. Polar Sensor Logger), or use an Apple Health export, whose HRV readings are
read automatically. With a week or more of nights, each night also gets a **recovery score** (0–100, 50 = normal *for
you*, compared with your previous 28 nights), with a plain-language reason. `openvitals trend
nights.csv` rescores saved results. Methods and validation are in [`docs/RECOVERY.md`](docs/RECOVERY.md).

From Python:

```python
from openvitals.infer import Stager, night_from_samples
result = Stager().stage(night_from_samples(unix_seconds, bpm))
print(result.summary); print(result.hypnogram())
```

## The model

The bundled stager is an ensemble of **10 small networks: 5 TCNs and 5 bidirectional GRUs**, each
trained with a different seed on 284 nights (all 253 BIDSleep heart-rate nights + Walch). Their
softmax outputs are averaged.

```
heart rate → 6 features per 30 s epoch
     (HR vs your night average, smoothed HR, HR variability over 2.5 / 5.5 min, change, hours into the night)
                     │
       ┌─────────────┴─────────────┐
   5 × TCN                     5 × GRU
   dilated convolutions,       reads the whole night
   sees ±1 h around each       forwards and backwards
   epoch (~38k params)         (~13k params)
       └─────────────┬─────────────┘
          average of 10 probability sets
                     │
          wake / light / deep / REM per epoch
```

- **Why two kinds:** they make different mistakes (TCNs did better with less data, GRUs with more),
  so averaging cancels some of each. Averaging 5 seeds removes most seed-to-seed variation.
- **Why heart rate only:** motion added nothing to these whole-night models, and it hurt transfer
  from home to lab nights. More heart-rate training nights helped far more (+0.05–0.07 kappa).
- **Training:** Adam, one night per step, class-weighted cross-entropy, early stopping on the kappa
  of held-out *people*.

### ONNX: one 1 MB file, no PyTorch

The ensemble ships as a single **ONNX** file (`src/openvitals/assets/stager_hr_4class.onnx`).
ONNX is a standard model format that runs anywhere onnxruntime does: Python, C++, Android, iOS,
web. Users don't need PyTorch (~1–3 GB), only onnxruntime (~15–50 MB). The graph standardises
inputs, runs all 10 networks and averages them, so the contract is just:

```python
import onnxruntime as ort
from openvitals import features
X = features.extract(night, "hr").X[None].astype("float32")     # (1, epochs, 6)
probs = ort.InferenceSession("stager_hr_4class.onnx").run(["probs"], {"features": X})[0]
```

Class names, feature order, scheme, version and data attribution (ODC-By) are stored in the file's
metadata. `scripts/validate_release.py` checks that the ONNX file reproduces the PyTorch benchmark
exactly (kappa 0.514 both ways). To build your own:
`openvitals export walch=data/walch --model tcn_hr --scheme 4class --out model.onnx`.

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

| `dreamt` | [PhysioNet DREAMT 2.2.0](https://physionet.org/content/dreamt/2.2.0/) | 100 / 100 | Empatica E4 wrist (HR, BVP, accel) + PSG ECG | PSG | **restricted** (data-use agreement); used only to *test*, never to train a shipped model |

Planned: MESA/SHHS via NSRR (data request). Model
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

`ens_hr` / `ens_motion_hr` average the softmax of a TCN and a GRU trained with the same seed;
`ens10_hr` is the shipped 5 TCN + 5 GRU ensemble. Ensembles export to one ONNX graph
(`openvitals export ... --model ens10_hr --jobs 4`).

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
  datasets/      walch, bidsleep, dreamt loaders (+ npz cache)
  features.py    per-epoch features
  models/        baselines
  splits.py      subject k-fold, LOSO, cross-dataset
  metrics.py     kappa, per-class, sleep summary, Bland–Altman
  models/sequence.py  whole-night TCN / bi-GRU + ONNX export
  benchmark.py   run + markdown/json report
  download.py    PhysioNet fetch with checksum verification
  featstore.py   per-subject feature files; streaming featurize
  infer.py       Stager: stage unlabelled nights with an exported ONNX model
  adapters.py    CSV, Apple Health and Gadgetbridge readers (HR, HRV, beat intervals); split_nights
  recovery.py    resting heart rate, beat cleaning, RMSSD, nightly HRV, recovery score
  beats.py       ECG R-peak and wrist PPG beat detection
  assets/        bundled release model (stager_hr_4class.onnx)
```

## Tests

```bash
uv run pytest
```

Tests use synthetic nights written in each dataset's on-disk format.

## License

Code: MIT (see `LICENSE`). The datasets keep their own licences (see the table above); this repository
contains no raw data. Not a medical device.
