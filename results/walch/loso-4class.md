# OpenVitals benchmark: loso / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 31 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|
| majority | 55.7% | 0.000 | 0.179 | 0.0% | 100.0% | 0.0% | 0.0% | 0.000 ± 0.000 |
| logreg_motion | 37.5% | 0.190 | 0.392 | 58.0% | 19.7% | 75.3% | 51.1% | 0.188 ± 0.087 |
| logreg_hr | 46.2% | 0.270 | 0.440 | 44.9% | 36.5% | 76.1% | 52.9% | 0.259 ± 0.102 |
| logreg_motion_hr | 49.3% | 0.305 | 0.484 | 59.2% | 38.0% | 74.8% | 58.2% | 0.297 ± 0.091 |
| gbdt_motion | 49.9% | 0.241 | 0.478 | 63.9% | 50.9% | 47.3% | 43.6% | 0.238 ± 0.102 |
| gbdt_hr | 54.1% | 0.326 | 0.506 | 50.0% | 50.8% | 57.1% | 62.4% | 0.321 ± 0.108 |
| gbdt_motion_hr | 60.0% | 0.376 | 0.555 | 59.6% | 63.0% | 50.2% | 58.5% | 0.364 ± 0.117 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| majority | +36.2 [+1.5, +70.9] | -20.3 [-50.2, +9.7] | +9.4% [-1.2%, +19.9%] | -13.0 [-30.6, +4.7] |
| logreg_motion | -23.4 [-87.0, +40.3] | +30.9 [-25.3, +87.2] | -5.6% [-22.6%, +11.4%] | -4.9 [-24.3, +14.6] |
| logreg_hr | -35.9 [-107.1, +35.4] | +48.0 [-13.2, +109.2] | -9.1% [-28.0%, +9.8%] | -9.3 [-30.1, +11.4] |
| logreg_motion_hr | -27.2 [-92.0, +37.5] | +34.2 [-22.7, +91.0] | -6.7% [-23.8%, +10.4%] | -4.3 [-24.9, +16.4] |
| gbdt_motion | -19.3 [-69.1, +30.5] | +19.8 [-26.5, +66.1] | -4.7% [-16.8%, +7.3%] | +2.0 [-17.4, +21.4] |
| gbdt_hr | -11.8 [-61.0, +37.3] | +11.7 [-29.2, +52.5] | -2.9% [-15.3%, +9.4%] | +3.0 [-17.2, +23.1] |
| gbdt_motion_hr | -11.9 [-59.7, +35.8] | +13.5 [-28.1, +55.0] | -3.0% [-14.9%, +9.0%] | +1.2 [-19.6, +22.0] |

Skipped (scheme not supported): cole_kripke
