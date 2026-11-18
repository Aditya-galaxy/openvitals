# Overnight resting heart rate, HRV and recovery score (OpenVitals 0.8.0)

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

### Real wrist PPG against ECG (DREAMT)

**Data:** DREAMT 2.2.0 (restricted, research use only), 20 sleep-lab patients with PSG ECG and an
Empatica E4 wristband recorded together. **Method:** `openvitals.beats` finds R-peaks in the ECG
(reference) and the steepest pulse upstroke in the wrist BVP waveform. Both go through the same
cleaning and RMSSD as above, in fixed 5-minute windows that PSG scored as asleep. Only windows with
a clean ECG count (76% of sleep windows; five subjects had mostly unusable ECG). Reproduce with
`scripts/validate_dreamt_hrv.py`; aggregate results are in `results/dreamt_hrv/`.

| level | n | median error | 90th pct error | ρ (ln RMSSD) | wrist ÷ ECG, geometric mean [95% LoA] |
|---|---|---|---|---|---|
| 5-min window | 752 windows (90% of clean-ECG windows had a wrist value) | 17.5% | 95.6% | 0.85 | – |
| night (median of windows) | 19 people | 16.2% | 78.9% | 0.88 | 1.28 [0.76, 2.13] |
| night, only the 14 people with clean ECG most of the night | 14 | 21.8% | – | 0.77 | 1.31 |

Mean heart rate per window agrees within 0.15 bpm (median), so beats are found correctly; the error
is in beat *timing*. The beat-to-beat variation of the ECG → wrist pulse delay is 36 ms (median;
IQR 28–56 ms) in the 14 clean-ECG subjects. Part of that is physiological (pulse transit time
changes with each breath), but it's far above the 5–15 ms simulated in test B2.

What this means:

- **Wrist HRV runs high.** On average the wrist reads RMSSD about 30% above ECG, because timing
  noise adds beat-to-beat variation. A typical nightly error is 15–20%, and one person in ten is off
  by about 80%.
- **It still ranks people and nights well** (ρ 0.85–0.88). For a personal baseline, which is how the
  recovery score uses HRV, a consistent overestimate cancels out. Absolute values shouldn't be
  compared with chest-strap or published norms.
- **Coverage is good when the band fits:** median 98% of sleep windows per person, but one subject
  had only 15%, presumably from a loose band or movement.
- Not tested: the E4's own beat-interval output. DREAMT stores it resampled and forward-filled, so
  individual beats can't be recovered reliably from it.

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

## Recovery score

A daily number from 0 to 100, where **50 means normal for you**. It's a transparent formula, not a
trained model, because there's no ground truth for "recovery".

1. For each measure, tonight is compared with **your previous 28 nights** (never later ones, so a
   past score never changes). A measure needs at least 7 earlier nights before it counts.
2. Each comparison is a standard score: (tonight − your average) ÷ your usual spread, clipped to
   ±3. Lower resting heart rate counts as better. The spread has a floor (HRV 0.05 ln-units, resting
   heart rate 1 bpm, sleep 20 min) so a very steady history can't trigger false alarms.
3. Weighted average: **HRV (ln RMSSD) 50%, resting heart rate 30%, total sleep time 20%**,
   re-weighted when a measure is missing. There's no score without HRV or resting heart rate.
4. Score = 50 + average × 50/3, so +3 → 100 and −3 → 0. Bands: **low < 35 ≤ normal < 65 ≤ high**.
5. Each score comes with plain words, e.g. *"HRV 29% below your average; resting HR +5.0 bpm vs your
   average; sleep −86 min vs your average"*.

`openvitals stage` scores every night in an export, and `openvitals trend nights.csv …` rescores
saved results (several exports can be combined; later files win per night).

**How to read it.** It answers "is tonight unusual *for me*?", not "am I healthy?". A low score
after a hard workout, late alcohol, illness or poor sleep is the expected signal. A single low
night matters less than several in a row. It isn't comparable between people, and it means nothing
for the first week (not enough history).

**What's checked.** Unit tests confirm a stress night (HRV −29%, resting heart rate +5 bpm, 86 min
less sleep) scores low with the right explanation, ordinary nights stay around 50, scores never use
future nights, and flat histories don't alarm. Its *usefulness* (does it track how people feel and
perform?) can only be shown on real multi-week data. None of the open datasets have enough nights
per person.

## Limitations

- **The HRV ground truth is 24-hour Holter**, not polysomnography. The "night" is the lowest-heart-
  rate 6 hours, and subjects are older adults in sinus rhythm (54 people). Arrhythmias such as
  atrial fibrillation make RMSSD meaningless.
- **Wrist-PPG HRV is tested on one device** (Empatica E4, 20 sleep-lab patients, above). It reads
  about 30% high, with 15–20% typical nightly error. Other watches' beat detection differs. Treat
  wrist HRV as a personal trend, not an absolute number.
- **Apple Health beat data is parsed best-effort.** Apple publishes no schema for `export.xml`,
  and the reader hasn't yet been checked against a real export. Unparseable readings fall back to
  Apple's own SDNN value, which is reported separately.

Data licences and citations: see `NOTICE`. `nsr2db` is used for validation only and is not part of
the package.
