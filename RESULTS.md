# Results

Reproduce with the commands in the README. Full tables, including per-night rows
and confusion matrices, are in `results/*/*.{md,json}`. Data is as described under
v0.1 below: Walch has motion + HR; BIDSleep is **HR only** locally, so every BIDSleep
number here is HR-only. [`notebooks/full_benchmark.ipynb`](notebooks/full_benchmark.ipynb)
fills in motion on Colab.

## v0.2: whole-night sequence models (2026-10-17)

`tcn_*` is a dilated residual TCN (receptive field ±126 epochs, about ±1 h) and `gru_*` is one
bidirectional GRU layer. Both are about 10–30k parameters, use the same per-epoch features as the
baselines, are early-stopped on held-out *training* subjects, and export to ONNX. Each number is a
**single seed**, with no confidence intervals yet.

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
