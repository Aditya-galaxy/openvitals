# OpenVitals benchmark: kfold / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|
| majority | 55.7% | 0.000 | 0.179 | 0.0% | 100.0% | 0.0% | 0.0% | 0.000 ± 0.000 |
| logreg_motion | 37.3% | 0.189 | 0.388 | 58.1% | 19.4% | 73.0% | 52.2% | 0.188 ± 0.084 |
| logreg_hr | 46.5% | 0.273 | 0.442 | 45.1% | 36.7% | 75.8% | 53.4% | 0.263 ± 0.104 |
| logreg_motion_hr | 49.6% | 0.306 | 0.485 | 59.3% | 38.6% | 73.2% | 58.8% | 0.300 ± 0.093 |
| gbdt_motion | 51.5% | 0.252 | 0.487 | 62.7% | 54.6% | 45.0% | 43.5% | 0.245 ± 0.110 |
| gbdt_hr | 53.4% | 0.309 | 0.494 | 48.8% | 51.5% | 52.3% | 60.7% | 0.303 ± 0.110 |
| gbdt_motion_hr | 60.4% | 0.371 | 0.551 | 58.3% | 65.6% | 44.7% | 57.8% | 0.359 ± 0.126 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| majority | +36.2 [+1.5, +70.9] | -20.3 [-50.2, +9.7] | +9.4% [-1.2%, +19.9%] | -13.0 [-30.6, +4.7] |
| logreg_motion | -24.5 [-89.2, +40.3] | +32.0 [-24.6, +88.6] | -5.9% [-23.7%, +11.9%] | -4.8 [-24.8, +15.1] |
| logreg_hr | -36.5 [-108.3, +35.2] | +48.5 [-13.4, +110.4] | -9.1% [-27.6%, +9.3%] | -9.1 [-30.1, +11.9] |
| logreg_motion_hr | -26.1 [-87.8, +35.5] | +33.3 [-20.5, +87.1] | -6.4% [-22.9%, +10.1%] | -4.4 [-24.7, +15.8] |
| gbdt_motion | -16.0 [-63.2, +31.2] | +16.7 [-28.1, +61.4] | -4.1% [-16.3%, +8.2%] | +1.8 [-19.7, +23.4] |
| gbdt_hr | -10.9 [-63.7, +41.9] | +11.6 [-31.8, +55.0] | -2.7% [-15.2%, +9.9%] | +1.9 [-18.0, +21.8] |
| gbdt_motion_hr | -8.8 [-54.3, +36.6] | +10.3 [-28.7, +49.3] | -2.2% [-13.5%, +9.1%] | +1.2 [-20.0, +22.3] |

Skipped (scheme not supported): cole_kripke
