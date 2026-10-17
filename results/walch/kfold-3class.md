# OpenVitals benchmark: kfold / 3class

- Data: walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | nrem sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|
| majority | 69.5% | 0.000 | 0.273 | 0.0% | 100.0% | 0.0% | 0.000 ± 0.000 |
| logreg_motion | 55.6% | 0.228 | 0.498 | 62.5% | 55.4% | 53.6% | 0.240 ± 0.112 |
| logreg_hr | 63.8% | 0.347 | 0.535 | 53.7% | 66.9% | 58.1% | 0.331 ± 0.137 |
| logreg_motion_hr | 66.6% | 0.387 | 0.580 | 62.9% | 67.8% | 64.0% | 0.377 ± 0.131 |
| gbdt_motion | 62.9% | 0.279 | 0.549 | 62.4% | 68.3% | 46.0% | 0.280 ± 0.113 |
| gbdt_hr | 68.0% | 0.390 | 0.580 | 49.9% | 71.4% | 64.4% | 0.382 ± 0.114 |
| gbdt_motion_hr | 72.3% | 0.433 | 0.626 | 59.2% | 78.4% | 57.8% | 0.420 ± 0.122 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| majority | +36.2 [+1.5, +70.9] | -20.3 [-50.2, +9.7] | +9.4% [-1.2%, +19.9%] | -13.0 [-30.6, +4.7] |
| logreg_motion | -33.0 [-103.9, +37.9] | +37.4 [-23.0, +97.9] | -8.1% [-28.1%, +11.9%] | -1.7 [-30.7, +27.2] |
| logreg_hr | -49.5 [-127.7, +28.7] | +58.3 [-10.1, +126.7] | -12.5% [-34.0%, +9.0%] | -6.0 [-27.7, +15.7] |
| logreg_motion_hr | -33.8 [-100.4, +32.8] | +39.6 [-17.6, +96.7] | -8.4% [-27.1%, +10.3%] | -3.5 [-23.8, +16.8] |
| gbdt_motion | -16.4 [-66.0, +33.2] | +17.3 [-28.4, +63.0] | -4.1% [-16.2%, +8.0%] | +1.8 [-20.0, +23.6] |
| gbdt_hr | -13.1 [-66.4, +40.1] | +13.4 [-30.3, +57.1] | -3.2% [-16.9%, +10.4%] | +2.6 [-17.2, +22.3] |
| gbdt_motion_hr | -10.5 [-58.5, +37.5] | +11.3 [-30.7, +53.3] | -2.6% [-14.6%, +9.3%] | +1.8 [-19.8, +23.4] |

Skipped (scheme not supported): cole_kripke
