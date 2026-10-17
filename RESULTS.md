# Baseline results (v0.1, 2026-10-17)

Reproduce with the commands in the README. Full tables, including per-night
rows and confusion matrices, are in `results/*/*.{md,json}`.

## Data actually used

| dataset | nights used | signals used | notes |
|---|---|---|---|
| Walch (sleep-accel 1.0.0) | 31 / 31 | motion + HR | via the AWS Open Data mirror; all checksums verified. Subject 7749105 has motion for only 13% of labelled epochs (only covered epochs are scored). |
| BIDSleep 1.0.1 | 253 / 253 | **HR only** | `motion.csv` is ~27 GB of the 27.9 GB, and PhysioNet served this machine at ~3 KB/s, with no mirror. HR + labels (~30 MB) only. |

Labels differ: Walch is lab PSG; BIDSleep is a Dreem 2 EEG headband with expert
correction, at home. So BIDSleep→Walch is both a device/setting shift and a
label-source shift.

## 4-class (wake / light / deep / REM), kappa

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

## 2-class (sleep / wake), Walch LOSO

| model | accuracy | kappa | wake sens | sleep sens |
|---|---|---|---|---|
| majority | 91.4% | 0.000 | 0% | 100% |
| cole_kripke | 83.8% | 0.305 | 58.8% | 86.2% |
| gbdt_motion | 87.5% | 0.419 | 67.7% | 89.3% |
| gbdt_motion_hr | 88.7% | 0.439 | 65.3% | 90.9% |

## Against the v1 targets

| target | v1 goal | now (best) | status |
|---|---|---|---|
| 4-stage kappa, cross-dataset | ≥ 0.55 | 0.314 (HR-only) | gap 0.24 |
| 4-stage kappa, within-dataset | – | 0.376 | – |
| 4-class accuracy | ≥ 70% | 60.0% | gap 10 pts |
| wake / deep sensitivity | ≥ 60% each | 59.6% / 50.2% | wake ≈ met, deep short |
| TST bias | within ±20 min | −11.9 min | met (limits of agreement ±48 min are wide) |

## What we learned

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

## Next steps, in expected order of impact

1. Sequence model over the whole night (bi-GRU or small TCN, ONNX-exportable)
   on the same features.
2. Accelerometer autocalibration, then re-test ENMO features.
3. Get BIDSleep motion from a faster route (the 5.9 GB zip from another network)
   for cross-dataset motion+HR results.
4. HR-derived variability features (BIDSleep HR is ~0.2 Hz, so true beat-level
   HRV needs DREAMT's PPG).
