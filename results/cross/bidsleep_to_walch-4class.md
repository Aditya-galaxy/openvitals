# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|
| majority | 55.7% | 0.000 | 0.179 | 0.0% | 100.0% | 0.0% | 0.0% | 0.000 ± 0.000 |
| logreg_hr | 46.6% | 0.277 | 0.439 | 39.1% | 35.9% | 72.9% | 60.4% | 0.265 ± 0.104 |
| gbdt_hr | 49.7% | 0.314 | 0.489 | 50.0% | 34.3% | 68.8% | 76.9% | 0.305 ± 0.092 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| majority | +36.2 [+1.5, +70.9] | -20.3 [-50.2, +9.7] | +9.4% [-1.2%, +19.9%] | -13.0 [-30.6, +4.7] |
| logreg_hr | -27.5 [-83.8, +28.8] | +39.1 [-7.7, +85.9] | -6.7% [-21.7%, +8.2%] | -8.6 [-28.7, +11.5] |
| gbdt_hr | -7.7 [-51.3, +35.9] | +7.4 [-29.0, +43.9] | -2.1% [-13.2%, +9.0%] | +3.2 [-16.2, +22.5] |
