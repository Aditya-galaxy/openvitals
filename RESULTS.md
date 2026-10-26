# Results

Reproduce with the commands in the README. Full tables, including per-night rows
and confusion matrices, are in `results/*/*.{md,json}`. Data is as described under
v0.1 below: Walch has motion + HR; BIDSleep is **HR only** locally, so every BIDSleep
number here is HR-only. [`notebooks/full_benchmark.ipynb`](notebooks/full_benchmark.ipynb)
fills in motion on Colab.

## v0.5: all 253 HR nights, pooled training, and ensembles (2026-10-26)

v0.4 concluded that more heart-rate nights beat motion. v0.5 tests that at full scale, locally,
from the feature stores (no new downloads): all **253 BIDSleep HR-only nights** (47 subjects) plus
the 31 Walch nights, HR only. Every number pools **5 seeds** with **95% subject-bootstrap CIs**.
Δ is a *paired* subject bootstrap. Comparisons across reports use the new
`openvitals compare A.json:MODEL B.json:MODEL`, which is paired over the test subjects both reports
share. Reports are in `results/v05/`.

**Pooled design.** We trained on Walch + BIDSleep pooled and evaluated with **subject-wise 5-fold
over all 78 subjects** (folds as `--seed 0`; each fold holds out 2–8 Walch and 7–14 BIDSleep
subjects). We didn't use leave-one-dataset-out: with only two datasets, it reduces to the two cross
splits we already have (BIDSleep → Walch below, and Walch → BIDSleep in v0.3), so it says nothing
about pooling. K-fold measures the model we would actually ship, trained on both cohorts. And
`merge --dataset` scores it on each cohort's subjects separately, so each half pairs with that
cohort's single-dataset numbers.

**Reproducibility check.** BIDSleep → Walch from the feature store reproduces v0.3, which was
run from raw files, **exactly**: Δ = 0.000 for every model and seed.

### 4-class kappa [95% CI], 5 seeds, HR only

| model | BIDSleep (253) → Walch | pooled 5-fold, Walch subjects | pooled 5-fold, BIDSleep subjects | pooled 5-fold, all 78 |
|---|---|---|---|---|
| gbdt_hr | 0.313 [0.279, 0.346] | 0.319 [0.285, 0.352] | 0.366 [0.343, 0.387] | 0.361 [0.339, 0.380] |
| tcn_hr | 0.482 [0.440, 0.523] | 0.488 [0.442, 0.530] | 0.512 [0.484, 0.538] | 0.509 [0.481, 0.532] |
| gru_hr | 0.488 [0.449, 0.526] | **0.497** [0.458, 0.535] | **0.531** [0.504, 0.556] | **0.527** [0.501, 0.549] |
| ens_hr (TCN + GRU, same seed) | 0.499 [0.457, 0.541] | – | – | – |
| ens10_hr (5 TCN + 5 GRU, seeds 0–4) | **0.514** [0.469, 0.558] | – | – | – |

`ens_hr` averages the softmax of the TCN and GRU trained with one seed. `ens10_hr` averages all 10
seed-0–4 models into one predictor, so it's a single run with no seed SD. Its members are exactly
the `tcn_hr`/`gru_hr` models above. On BIDSleep → Walch, accuracy is 66.6% [63.6, 69.5] (`ens10_hr`),
and TST bias is −0.3 min [−38.3, +37.7], the narrowest limits of agreement so far.

### Paired Δ kappa [95% CI], same test subjects

| question | comparison | TCN | GRU | gbdt |
|---|---|---|---|---|
| 253 vs 47 BIDSleep nights (v0.4 replicated) | BIDSleep→Walch: v0.5 − v0.4 | **+0.048** [+0.033, +0.063] | **+0.068** [+0.051, +0.086] | +0.022 [+0.008, +0.035] |
| add Walch to the 253 nights | Walch subjects: pooled − BIDSleep→Walch | +0.006 [−0.005, +0.016] | +0.009 [−0.005, +0.024] | +0.006 [−0.000, +0.013] |
| add 253 BIDSleep nights to Walch | Walch subjects: pooled − v0.3 Walch 5-fold | +0.009 [−0.017, +0.036] | **+0.073** [+0.047, +0.096] | +0.009 [−0.017, +0.036] |
| add Walch to BIDSleep | BIDSleep subjects: pooled − v0.3 BIDSleep 5-fold | −0.001 [−0.005, +0.004] | +0.003 [−0.004, +0.011] | +0.002 [−0.001, +0.005] |
| HR vs best in-domain model | Walch subjects: pooled gru_hr − v0.3 LOSO tcn_motion_hr | – | −0.001 [−0.022, +0.020] | – |

| ensembles (BIDSleep → Walch) | Δ kappa [95% CI] | P(Δ ≤ 0) |
|---|---|---|
| ens_hr − gru_hr | +0.011 [+0.004, +0.020] | 0.001 |
| ens10_hr − gru_hr | **+0.026 [+0.017, +0.036]** | 0.000 |
| ens10_hr − ens_hr | +0.014 [+0.009, +0.019] | 0.000 |

### What we learned

1. **"More HR nights beat motion" holds at full scale.** The 253-night models replicate v0.4's
   +0.048 (TCN) and +0.068 (GRU) over 47 nights, and v0.4's best motion gain was +0.013, unresolved.
2. **Walch subjects plateau at about 0.49–0.50 kappa, however the model is trained.**
   - In-domain Walch LOSO with motion: 0.498.
   - BIDSleep → Walch: 0.488.
   - Pooled, with 179–217 BIDSleep + 23–29 Walch training nights per fold: 0.497.

   Adding Walch's own nights to 253 BIDSleep nights doesn't resolve a gain (+0.009
   [−0.005, +0.024]), and HR-only pooled ties the best motion + HR in-domain model (Δ −0.001). With 31
   lab nights and these HR features, the remaining error looks like a property of the test cohort
   (its PSG labels and HR signal) rather than missing training data or domain shift.
3. **The GRU is the model that benefits from pooling.** Added to Walch, BIDSleep lifts the GRU by
   +0.073 on Walch subjects but the TCN by only +0.009. The GRU also wins every pooled column (+0.018
   [+0.014, +0.022] over the TCN on all 78 subjects), which confirms v0.3 point 4.
   `gru_hr` trained on the pooled set is the model to export.
4. **The only resolved gain on the cross split came from ensembling.** A TCN + GRU pair adds +0.011,
   and averaging 10 seed-and-architecture members adds +0.026. That makes `ens10_hr` the best
   cross-dataset result so far (0.514). It costs 10× the training and inference of one model, but the
   members are small (about 10–30k parameters each).
5. **BIDSleep subjects don't benefit from Walch either.** 23–29 extra nights on top of ~180–220 per fold
   change nothing (Δ ≤ 0.003).

### Against the v1 targets

| target | v1 goal | v0.5 best | status |
|---|---|---|---|
| 4-stage kappa, cross-dataset | ≥ 0.55 | **0.514** [0.469, 0.558] (ens10_hr, BIDSleep 253 → Walch) | **not met**: gap 0.036. The CI's upper end reaches 0.55, but the point estimate is short. |
| 4-stage kappa, within / pooled | – | 0.531 [0.504, 0.556] (gru_hr, BIDSleep subjects, pooled) | – |
| 4-class accuracy | ≥ 70% | 67.0% [65.1, 68.8] (gru_hr pooled, BIDSleep subjects) | not met |
| TST bias | within ±20 min | −0.3 min, limits of agreement ±38 min (ens10_hr, → Walch) | met; limits of agreement still wide |

Progress on cross-dataset kappa: 0.314 (v0.1) → 0.483 (v0.2) → 0.488 (v0.3) → **0.514** (v0.5).
More BIDSleep-style HR nights won't close the last 0.036. Pooling and the full 253 nights are both
used up, and Walch scores the same in-domain.

### Next steps (revised)

1. **A new test cohort with denser HR or PPG.** DREAMT (credentialed, PPG + PSG, 100 subjects) can
   test whether the ~0.50 Walch plateau comes from the cohort (31 nights, coarse HR) or from the
   method. Beat-level HRV features need PPG.
2. **Ship pooled `gru_hr`, and export the ensemble.** Make `ens*` ONNX-exportable as one graph
   that averages softmax, then export it trained on the pooled set.
3. **A cheaper ensemble.** Check whether 3 + 3 members keep most of the +0.026.

---

## v0.4: motion + HR on both datasets (2026-10-26)

BIDSleep **with motion** was featurized on Google Cloud (`cloud/gcp/`). PhysioNet throttles
downloads to about 25 KB/s per connection everywhere we tried, so this uses **1 night per subject
(all 47 subjects)**. That's ~5.2 GB in ~18 h with 8 parallel workers. Motion covers 92% of labelled
epochs. Every number pools **5 seeds** with **95% subject-bootstrap CIs**, and Δ is a *paired*
bootstrap (method as in v0.3). Reports are in `results/v04/`.

### 4-class kappa [95% CI], 5 seeds, BIDSleep = 47 subjects × 1 night

| model | BIDSleep 5-fold | BIDSleep → Walch | Walch → BIDSleep |
|---|---|---|---|
| gbdt_hr | 0.335 [0.300, 0.365] | 0.291 [0.261, 0.319] | 0.312 [0.276, 0.345] |
| gbdt_motion_hr | 0.347 [0.312, 0.380] | 0.324 [0.294, 0.352] | 0.299 [0.266, 0.329] |
| tcn_hr | **0.476** [0.424, 0.520] | 0.434 [0.390, 0.475] | **0.431** [0.385, 0.472] |
| tcn_motion_hr | 0.467 [0.418, 0.511] | **0.447** [0.404, 0.487] | 0.411 [0.367, 0.450] |
| gru_hr | 0.448 [0.396, 0.495] | 0.420 [0.378, 0.458] | 0.417 [0.373, 0.459] |
| gru_motion_hr | 0.446 [0.397, 0.489] | 0.425 [0.381, 0.465] | 0.381 [0.333, 0.424] |

### Does motion help? (paired Δ kappa, motion + HR − HR only)

| split | TCN | GRU | gbdt |
|---|---|---|---|
| BIDSleep 5-fold | −0.009 [−0.024, +0.007] | −0.002 | +0.013 [−0.005, +0.031] |
| BIDSleep → Walch | +0.013 [−0.003, +0.027] (P = 0.06) | +0.005 | +0.033 [+0.014, +0.051] |
| Walch → BIDSleep | **−0.020 [−0.041, −0.000]** | **−0.037** | −0.013 [−0.031, +0.005] |

(GRU and gbdt rows are read from each report's reference comparisons. GRU's Δ is the difference of the
two Δs vs `tcn_hr`, so it's indicative only.)

### Do more training nights help? (paired Δ, same Walch test subjects, HR only)

BIDSleep → Walch trained on **253 nights** (v0.3) vs **47 nights** (v0.4):

| model | 253 nights | 47 nights | Δ [95% CI] |
|---|---|---|---|
| tcn_hr | 0.482 | 0.434 | **+0.048 [+0.033, +0.063]** |
| gru_hr | 0.488 | 0.420 | **+0.068 [+0.051, +0.086]** |
| gbdt_hr | 0.313 | 0.291 | +0.022 [+0.008, +0.035] |

### What we learned

1. **Motion adds no kappa, and it hurts transfer from lab to home.** With motion on both datasets,
   no split shows a resolved gain for the sequence models. Trained on Walch (lab sleep studies) and
   tested on BIDSleep (home), motion makes the TCN *worse* (−0.020) and the GRU worse still.
   Motion shifts predictions toward wake: wake sensitivity 51% → 64% (TCN, Walch→BIDSleep), but deep
   and REM drop, and total-sleep-time bias grows from −7 to −27 min. Wrist movement during sleep
   differs between a lab bed and home, and ENMO is calibration-dependent (v0.1, point 3).
   Baselines with no night context still gain from motion (gbdt +0.033 BIDSleep→Walch).
2. **More nights beat more signals.** Training on 253 HR-only nights instead of 47 adds +0.05 to +0.07
   kappa, which is more than motion ever did. HR + labels for all 253 nights is only ~30 MB, while
   motion is ~27 GB.
3. **The sequence models still beat the baselines everywhere**, by +0.10 to +0.14 kappa (all CIs
   exclude 0).
4. **TCN vs GRU tracks training-set size again.** On 47 BIDSleep nights the TCN wins (+0.028
   [+0.014, +0.041] on 5-fold). On 253 nights the GRU won (v0.3), consistent with the GRU needing
   more data.

### Against the v1 targets

The best cross-dataset 4-class kappa is still **0.488** (gru_hr, BIDSleep → Walch, 253 HR nights, v0.3).
The 0.55 target is not met. v0.4 doesn't beat it, because it trains on 5× fewer nights.

### Next steps (revised)

1. **Drop motion from the main model; scale heart-rate nights instead.** Pool all 253 BIDSleep HR
   nights with Walch for training (cheap), and add DREAMT (credentialed, PPG + PSG, 100 subjects).
2. Keep `*_motion_hr` as an option for wake-sensitive uses, after per-device accelerometer
   recalibration.
3. Night-level TST limits of agreement (±45–60 min) are still the weakest number.

---

## v0.3: five seeds and subject-bootstrap confidence intervals (2026-10-17)

Every 4-class number below pools **5 training seeds** (0–4) on fixed folds. Brackets are **95% CIs
from resampling test subjects** (2,000 draws). Subjects are the independent unit, not epochs or
nights. "Seed SD" is the spread of pooled kappa across seeds. **Δ** is a *paired* subject bootstrap:
both models are scored on the same resampled subjects, which removes between-subject variance, so it
can resolve gains far smaller than the width of each model's own CI. Merged reports are in
`results/v03/`, and per-seed raw reports in `results/v03/parts/`.

### 4-class kappa [95% CI], 5 seeds

| model | Walch 5-fold | Walch LOSO | BIDSleep 5-fold (HR) | BIDSleep → Walch (HR) | Walch → BIDSleep (HR) |
|---|---|---|---|---|---|
| gbdt_hr | 0.310 [0.271, 0.348] | 0.326 [0.286, 0.365] | 0.364 [0.341, 0.385] | 0.313 [0.279, 0.346] | 0.301 [0.280, 0.320] |
| gbdt_motion_hr | 0.370 [0.329, 0.410] | 0.374 [0.334, 0.413] | – | – | – |
| tcn_hr | **0.479** [0.432, 0.522] | 0.488 [0.442, 0.530] | 0.513 [0.483, 0.540] | 0.482 [0.440, 0.523] | **0.434** [0.409, 0.458] |
| tcn_motion_hr | 0.472 [0.427, 0.513] | **0.498** [0.459, 0.535] | – | – | – |
| gru_hr | 0.424 [0.375, 0.470] | – | **0.528** [0.498, 0.554] | **0.488** [0.449, 0.526] | 0.404 [0.375, 0.430] |
| gru_motion_hr | 0.444 [0.393, 0.488] | – | – | – | – |

Seed SD is 0.001–0.004 for gbdt and 0.004–0.016 for the sequence models. GRU wasn't run on Walch LOSO
(~65 min per seed). BIDSleep is HR-only locally.

### Paired differences (Δ kappa [95% CI], P(Δ ≤ 0))

| question | split | Δ kappa | P(Δ ≤ 0) | verdict |
|---|---|---|---|---|
| sequence model vs best baseline | Walch LOSO: tcn_hr − gbdt_motion_hr | +0.114 [+0.072, +0.157] | 0.000 | real |
| | BIDSleep→Walch: gru_hr − gbdt_hr | +0.175 [+0.149, +0.202] | 0.000 | real |
| | Walch→BIDSleep: tcn_hr − gbdt_hr | +0.133 [+0.120, +0.147] | 0.000 | real |
| does motion help a sequence model? | Walch LOSO: tcn_motion_hr − tcn_hr | +0.010 [−0.013, +0.033] | 0.189 | **not resolved** |
| | Walch 5-fold: tcn_motion_hr − tcn_hr | −0.007 [−0.029, +0.015] | 0.751 | not resolved |
| TCN vs GRU | Walch 5-fold: gru_hr − tcn_hr | −0.054 [−0.079, −0.030] | 1.000 | TCN better |
| | Walch→BIDSleep: gru_hr − tcn_hr | −0.031 [−0.040, −0.021] | 1.000 | TCN better |
| | BIDSleep 5-fold: tcn_hr − gru_hr | −0.015 [−0.021, −0.009] | 1.000 | **GRU better** |
| | BIDSleep→Walch: tcn_hr − gru_hr | −0.006 [−0.021, +0.009] | 0.767 | not resolved |
| does motion help a baseline? | Walch LOSO: gbdt_hr − gbdt_motion_hr | −0.048 [−0.074, −0.017] | 0.999 | motion helps gbdt |

### What changed from v0.2

1. **The v0.2 headline was the luckiest seed.** Walch LOSO tcn_hr seeds score 0.511, 0.497, 0.478,
   0.479 and 0.475. v0.2 reported seed 0 (0.511), and the 5-seed estimate is 0.488. Single-seed numbers
   overstated the best model by about 0.02.
2. **Sequence models beat the baselines, clearly.** Δ is +0.11 to +0.18 kappa in every split, with CIs
   far from zero. This is the one big, solid result.
3. **Motion adds no measurable kappa once the model sees the whole night.** Paired Δ is +0.010
   [−0.013, +0.033] on LOSO, so any true effect is below ~0.03. For gbdt, which has no night context,
   motion does help (+0.048). Motion may still help with wake specifically: tcn_motion_hr's wake
   sensitivity is 69% [62%, 77%] against tcn_hr's 59% [52%, 66%], and it's steadier across seeds
   (SD 0.007 vs 0.015). Those intervals touch, so treat it as a lead, not a result.
4. **"TCN vs GRU" depends on training-set size, and now that's resolvable.** On 31 Walch nights the TCN
   wins (by 0.03–0.05). On 253 BIDSleep nights the GRU wins by a small but resolved +0.015
   [+0.009, +0.021], a gap far narrower than either model's CI (about ±0.03).
5. **Cross-dataset still trails within-dataset.** BIDSleep→Walch 0.488 vs Walch LOSO 0.488–0.498 is
   about equal. Walch→BIDSleep 0.434 vs BIDSleep 5-fold 0.528 is a real gap: training on 31 nights
   doesn't transfer as well as training on 253.

### Against the v1 targets (5-seed estimates)

| target | v1 goal | v0.3 best | status |
|---|---|---|---|
| 4-stage kappa, cross-dataset | ≥ 0.55 | 0.488 [0.449, 0.526] (gru_hr, BIDSleep→Walch) | **not met**: the CI excludes 0.55 |
| 4-class accuracy | ≥ 70% | 66.7% [64.5%, 68.5%] (gru_hr, BIDSleep 5-fold) | not met: the CI excludes 70% |
| wake / deep sensitivity | ≥ 60% each | wake 69% [62, 77], deep 73% [67, 79] (tcn_motion_hr, Walch LOSO) | **met** |
| TST bias | within ±20 min | −9.6 min (tcn_hr, Walch LOSO); −5.5 (gru_hr, BIDSleep) | met, but limits of agreement are about ±50–60 min |

### Next steps

1. **BIDSleep motion on Colab** ([`notebooks/full_benchmark.ipynb`](notebooks/full_benchmark.ipynb)), then
   test the wake-sensitivity lead from point 3 on 253 nights, where the CIs are about half as wide.
2. **Pool both datasets for training.** The GRU benefits from more data, and cross-dataset is the
   target that's still missed.
3. **Night-level agreement.** TST limits of agreement of about ±55 min are the weakest number here.
   Consider a night-level calibration head, or report TST from an ensemble of seeds.

---

## v0.2: whole-night sequence models (2026-10-17)

`tcn_*` is a dilated residual TCN (receptive field ±126 epochs, about ±1 h) and `gru_*` is one
bidirectional GRU layer. Both are about 10–30k parameters, use the same per-epoch features as the
baselines, are early-stopped on held-out *training* subjects, and export to ONNX. Each number is a
**single seed** (seed 0) with no CIs, so it's superseded by the 5-seed estimates in v0.3 above.

### 4-class kappa (wake / light / deep / REM)

| model | Walch 5-fold | Walch LOSO | BIDSleep 5-fold (HR) | BIDSleep → Walch (HR) | Walch → BIDSleep (HR) |
|---|---|---|---|---|---|
| best v0.1 baseline | 0.371 | 0.376 | 0.364 | 0.314 | 0.302 |
| tcn_hr | 0.484 | **0.511** | 0.507 | 0.455 | **0.426** |
| tcn_motion_hr | **0.486** | 0.500 | – | – | – |
| gru_hr | 0.410 | 0.432 | **0.528** | **0.483** | 0.399 |
| gru_motion_hr | 0.430 | 0.452 | – | – | – |

Best baseline is gbdt_motion_hr on Walch and gbdt_hr in the HR-only columns.

Best within-dataset result (Walch LOSO, tcn_hr): accuracy 67.6%, macro F1 0.646. Sensitivity: wake 60.7%,
light 62.8%, deep 76.3%, REM 77.0%. TST bias −9.2 min [−53.0, +34.7]; WASO +9.7 min [−30.8, +50.2].

### 3-class and 2-class kappa

| model | Walch 5-fold 3c | BIDSleep→Walch 3c | Walch→BIDSleep 3c | Walch LOSO 2c | BIDSleep→Walch 2c | Walch→BIDSleep 2c |
|---|---|---|---|---|---|---|
| best v0.1 baseline | 0.433 | 0.357 | 0.341 | 0.439 | 0.283 | 0.351 |
| tcn_hr | **0.528** | 0.538 | **0.453** | 0.501 | 0.354 | 0.401 |
| tcn_motion_hr | 0.526 | – | – | **0.532** | – | – |
| gru_hr | 0.473 | **0.552** | 0.413 | – | **0.409** | **0.447** |

### Against the v1 targets

| target | v1 goal | v0.1 | v0.2 (best) | status |
|---|---|---|---|---|
| 4-stage kappa, cross-dataset | ≥ 0.55 | 0.314 | 0.483 (gru_hr, BIDSleep→Walch) | gap 0.07 |
| 4-stage kappa, within-dataset | – | 0.376 | 0.528 (gru_hr, BIDSleep) | – |
| 4-class accuracy | ≥ 70% | 60.0% | 67.6% | gap 2.4 pts |
| wake / deep sensitivity | ≥ 60% each | 59.6% / 50.2% | 60.7% / 76.3% | **met** |
| TST bias | within ±20 min | −11.9 min | −9.2 min | **met** (limits of agreement still ±44 min) |

### What we learned

1. **Whole-night context is the big win.** Sequence models gain +0.12 to +0.17 kappa over the best
   baseline in every split, and deep-sleep sensitivity rises from about 50% to 70–76%.
2. **With context, motion barely helps.** On Walch LOSO, HR-only scores 0.511 and motion + HR scores 0.500.
   HR-only models are viable, which suits devices whose accelerometer data is poor or unavailable. Caveat:
   the motion features still use uncalibrated ENMO (see v0.1, point 3), so motion may be undervalued.
3. **The GRU wants more data, and the TCN is cheaper.** Trained on 31 Walch nights, the TCN wins.
   Trained on 253 BIDSleep nights, the GRU wins (0.528 within, 0.483 cross). The TCN trains about 10×
   faster (7 vs 74 min for Walch LOSO).
4. **Cross-dataset transfer works better from the larger cohort.** BIDSleep→Walch (0.483) is close to
   Walch's own LOSO (0.511), but Walch→BIDSleep (0.426) is well below BIDSleep's own 5-fold (0.528).
5. **Night totals are unbiased but noisy.** TST bias is under 10 min, but the 95% limits of agreement are
   about ±45 min, which is too wide to trust one night's number.

### Next steps

1. Run the notebook on Colab to get BIDSleep motion, then rerun the motion + HR columns.
2. Use several seeds and bootstrap confidence intervals over subjects, so 0.01–0.02 differences stop
   looking meaningful.
3. Train on both datasets pooled, which is the model to export, and keep the cross-dataset numbers as its
   validation.
4. Recalibrate the accelerometer, then re-test whether motion adds anything to the sequence models.

---

## v0.1: baselines (2026-10-17)

### Data actually used

| dataset | nights used | signals used | notes |
|---|---|---|---|
| Walch (sleep-accel 1.0.0) | 31 / 31 | motion + HR | via the AWS Open Data mirror; all checksums verified. Subject 7749105 has motion for only 13% of labelled epochs (only covered epochs are scored). |
| BIDSleep 1.0.1 | 253 / 253 | **HR only** | `motion.csv` is ~27 GB of the 27.9 GB, and PhysioNet served this machine at ~3 KB/s, with no mirror. HR + labels (~30 MB) only. |

Labels differ: Walch is lab PSG; BIDSleep is a Dreem 2 EEG headband with expert
correction, at home. So BIDSleep→Walch is both a device/setting shift and a
label-source shift.

### 4-class (wake / light / deep / REM), kappa

| model | Walch 5-fold | Walch LOSO | BIDSleep 5-fold | BIDSleep → Walch | Walch → BIDSleep |
|---|---|---|---|---|---|
| majority | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| logreg_hr | 0.273 | 0.270 | 0.290 | 0.277 | 0.260 |
| gbdt_hr | 0.309 | 0.326 | 0.364 | **0.314** | 0.302 |
| logreg_motion_hr | 0.306 | 0.305 | – | – | – |
| gbdt_motion | 0.252 | 0.241 | – | – | – |
| **gbdt_motion_hr** | **0.371** | **0.376** | – | – | – |

Best (Walch LOSO, gbdt_motion_hr): accuracy 60.0%, macro F1 0.555. Sensitivity:
wake 59.6%, light 63.0%, deep 50.2%, REM 58.5%. TST bias −11.9 min
[−59.7, +35.8]; WASO +13.5 min [−28.1, +55.0].

### 2-class (sleep / wake), Walch LOSO

| model | accuracy | kappa | wake sens | sleep sens |
|---|---|---|---|---|
| majority | 91.4% | 0.000 | 0% | 100% |
| cole_kripke | 83.8% | 0.305 | 58.8% | 86.2% |
| gbdt_motion | 87.5% | 0.419 | 67.7% | 89.3% |
| gbdt_motion_hr | 88.7% | 0.439 | 65.3% | 90.9% |

### Against the v1 targets

| target | v1 goal | now (best) | status |
|---|---|---|---|
| 4-stage kappa, cross-dataset | ≥ 0.55 | 0.314 (HR-only) | gap 0.24 |
| 4-stage kappa, within-dataset | – | 0.376 | – |
| 4-class accuracy | ≥ 70% | 60.0% | gap 10 pts |
| wake / deep sensitivity | ≥ 60% each | 59.6% / 50.2% | wake ≈ met, deep short |
| TST bias | within ±20 min | −11.9 min | met (limits of agreement ±48 min are wide) |

### What we learned

1. **HR-only transfers.** gbdt_hr trained on BIDSleep scores 0.314 on Walch,
   matching Walch-trained 0.309–0.326, despite the label-source and setting
   shift. More nights from a different cohort is as good as in-cohort data.
2. **Motion + HR beats either alone** (0.376 vs 0.326 HR, 0.241 motion,
   4-class LOSO). Cross-dataset motion results need BIDSleep motion.
3. **ENMO depends on device calibration.** On Walch the resting ENMO floor
   varies 0–1.1 (log units) between subjects. Cole-Kripke on ENMO scored kappa
   0.10; on magnitude SD 0.29–0.31. ENMO features need per-night
   autocalibration (van Hees-style) before they can be trusted across devices.
4. **Balanced class weights cost accuracy.** Most balanced models sit below the
   majority baseline's accuracy (they trade light-sleep recall for
   wake/deep/REM). Kappa and per-class sensitivity are the honest metrics.
5. **Per-epoch models ignore night structure.** These models only see
   ±10-epoch windows. Published sequence models (e.g. SLAMSS LSTM on MESA,
   70–72% 4-class) suggest sequence context is the next big gain.

### Next steps, in expected order of impact

1. Sequence model over the whole night (bi-GRU or small TCN, ONNX-exportable)
   on the same features.
2. Accelerometer autocalibration, then re-test ENMO features.
3. Get BIDSleep motion from a faster route (the 5.9 GB zip from another network)
   for cross-dataset motion+HR results.
4. HR-derived variability features (BIDSleep HR is ~0.2 Hz, so true beat-level
   HRV needs DREAMT's PPG).
