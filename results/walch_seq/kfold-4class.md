# OpenVitals benchmark: kfold / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|
| tcn_hr | 65.9% | 0.484 | 0.619 | 54.4% | 62.4% | 69.4% | 77.1% | 0.471 ± 0.145 |
| tcn_motion_hr | 66.0% | 0.486 | 0.633 | 64.7% | 61.8% | 66.3% | 77.0% | 0.471 ± 0.137 |
| gru_hr | 59.7% | 0.410 | 0.560 | 54.5% | 54.2% | 64.4% | 72.9% | 0.399 ± 0.140 |
| gru_motion_hr | 61.5% | 0.430 | 0.593 | 65.0% | 55.6% | 66.5% | 71.9% | 0.413 ± 0.142 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| tcn_hr | -7.0 [-48.7, +34.8] | +6.5 [-32.2, +45.3] | -2.0% [-12.2%, +8.2%] | +1.6 [-17.8, +21.0] |
| tcn_motion_hr | -8.5 [-49.1, +32.1] | +7.7 [-29.1, +44.5] | -2.7% [-13.6%, +8.2%] | +1.5 [-16.9, +19.8] |
| gru_hr | -21.8 [-77.4, +33.8] | +19.3 [-22.6, +61.3] | -5.2% [-20.5%, +10.1%] | +3.5 [-29.5, +36.5] |
| gru_motion_hr | -14.4 [-75.6, +46.8] | +12.2 [-36.9, +61.3] | -3.4% [-17.3%, +10.4%] | +3.7 [-23.3, +30.8] |
