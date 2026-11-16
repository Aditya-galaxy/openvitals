# Overnight resting heart rate and HRV (OpenVitals 0.7.0)

**Not a medical device.** Wellness and research use only.

`openvitals stage` reports these for each night next to the sleep stages:

| metric | needs | method |
|---|---|---|
| **Resting heart rate** | the heart rate `stage` already uses | lowest 5-minute mean heart rate while asleep (a window counts if ≥ 8 of its 10 epochs are asleep and it holds ≥ 3 samples) |
| **Average sleeping heart rate** | same | mean of all heart-rate samples in sleep epochs |
| **HRV (RMSSD)** | **beat-to-beat intervals**: a chest-strap export (`--ibi beats.csv`) or Apple Health HRV readings | 5-minute windows while asleep with ≥ 80% clean beats; nightly value = median window. For Apple, RMSSD per ~1-minute reading, median of readings |

**Why HRV needs beat-to-beat data.** HRV is the variation *between consecutive heartbeats*.
Heart rate logged every few seconds (what most exports contain) is an average over several beats,
which removes exactly that variation. So HRV is only reported when beat intervals are available.

**Beat cleaning.** An interval is dropped if it's outside 300–2000 ms (30–200 bpm) or more than
20% away from the median of the surrounding 11 intervals. That catches missed beats (one long
interval), extra beats (two short ones) and ectopic short-long pairs. RMSSD uses only *successive*
pairs of clean beats, so a removed beat never creates a fake large difference.

## Validation

Reproduce with `scripts/validate_hrv.py` and `scripts/validate_rhr.py`.

### HRV against ECG ground truth

**Data:** PhysioNet Normal Sinus Rhythm RR Interval Database (`nsr2db`, ODC-By 1.0): 54 adults
(28–76 years), ~24 h Holter ECG, every beat annotated and manually reviewed.
**Reference:** RMSSD from normal-to-normal intervals only, median of 5-min windows over the 6 hours
with the lowest heart rate. The headers carry no time of day, so this is a night proxy.
Reference median 30.7 ms (IQR 22.0–39.9).
**Error** = |estimate − reference| / reference per person. **ρ** = Spearman rank correlation of
ln RMSSD across people, which shows whether it still tells people apart.

| condition | median error | 90th pct error | ρ |
|---|---|---|---|
| **A. Real ectopic beats** in the recordings, no cleaning | 3.4% | 58.4% | 0.94 |
| A. Real ectopic beats, **cleaned** | **2.2%** | **15.5%** | 0.94 |
| **B1. Injected artifacts** (missed / extra / ectopic) 1%, no cleaning | 229.5% | 452.1% | 0.62 |
| B1. 1% artifacts, **cleaned** | **2.0%** | 15.0% | 0.93 |
| B1. 5% artifacts, no cleaning | 640.5% | 1085.6% | 0.47 |
| B1. 5% artifacts, **cleaned** | **3.9%** | 12.5% | 0.94 |
| **B2. Beat-timing jitter** 5 ms, cleaned | 5.3% | 16.2% | 0.93 |
| B2. 10 ms jitter, cleaned | 14.0% | 41.3% | 0.94 |
| B2. 15 ms jitter, cleaned | 30.3% | 73.6% | 0.93 |
| **C. Apple-style**: 3 × 1-min readings, whole-number bpm | 14.0% | 44.4% | 0.73 |
| C. 6 × 1-min readings, whole-number bpm | 9.6% | 22.1% | 0.90 |
| C. 12 × 1-min readings, whole-number bpm | 7.9% | 21.9% | 0.93 |

What this means:

- **Cleaning is essential, and it works.** Even 1% bad beats inflate raw RMSSD several-fold. After
  cleaning, error stays around 2–4% at up to 5% artifacts.
- **Beat-timing precision is the real limit.** Cleaning can't remove small errors in *when* each
  beat was detected: 5 ms costs about 5%, 10 ms about 14%, 15 ms about 30%. Chest straps (ECG) are
  precise. Wrist optical sensors vary, so wrist-derived HRV is only as good as the device's beat
  detection.
- **Sparse readings work moderately.** Apple Watch takes a few ~1-minute HRV readings per night.
  With 6 readings the nightly value is within about 10% (median), and ranking across people holds
  (ρ 0.90). With 3 readings it's noticeably worse. Rounding to whole bpm, as Apple stores beats,
  costs little.

### Resting heart rate

**1. Does staging error matter?** Walch (31 lab nights). Resting heart rate using sleep from
polysomnography vs from the model's predicted stages, with a model that never saw Walch.

| heart-rate sampling | metric | typical value | predicted − PSG: bias [95% limits] |
|---|---|---|---|
| ~5 s | resting HR (lowest 5 min) | 58.2 bpm | +0.08 bpm [−1.28, +1.43] |
| ~5 s | average sleeping HR | 64.9 bpm | +0.12 bpm [−0.59, +0.83] |
| 1 per 60 s | resting HR (lowest 5 min) | 57.2 bpm | +0.15 bpm [−1.19, +1.48] |
| 1 per 60 s | average sleeping HR | 65.2 bpm | +0.07 bpm [−0.58, +0.71] |

Staging mistakes move resting heart rate by well under 1 bpm, including with once-a-minute heart
rate. Its value is a little lower at 60 s sampling (57.2 vs 58.2 bpm): with fewer samples, the
lowest of many noisy 5-minute means is slightly lower.

**2. Night-to-night repeatability.** BIDSleep: 47 people, 253 home nights, sleep from the EEG
headband labels.

| metric | within-person SD | ICC(1,1) |
|---|---|---|
| resting HR (lowest 5 min) | 3.4 bpm | 0.69 |
| average sleeping HR | 3.6 bpm | 0.70 |

About 70% of the variation is between people, so a single night's value reliably reflects the
person. Night-to-night swings of about 3–4 bpm are normal (some of it is real: alcohol, late meals,
illness, training), so compare against your own recent baseline, not a single night.

## Limitations

- **The HRV ground truth is 24-hour Holter**, not polysomnography. The "night" is the lowest-heart-
  rate 6 hours, and subjects are older adults in sinus rhythm (54 people). Arrhythmias such as
  atrial fibrillation make RMSSD meaningless.
- **No real wrist-PPG HRV validation yet.** Test B2 simulates beat-timing jitter. Measuring a real
  wrist sensor needs a dataset with PPG and ECG together (e.g. DREAMT, once access is granted).
- **Apple Health beat data is parsed best-effort.** Apple publishes no schema for `export.xml`,
  and the reader hasn't yet been checked against a real export. Unparseable readings fall back to
  Apple's own SDNN value, which is reported separately.
- **No recovery score yet.** A personal-baseline recovery score (tonight's HRV and resting heart
  rate against your last few weeks) is the next step. It can't be validated against ground truth,
  so it will be a transparent formula, not an "accuracy" claim.

Data licences and citations: see `NOTICE`. `nsr2db` is used for validation only and is not part of
the package.
