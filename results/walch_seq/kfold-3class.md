# OpenVitals benchmark: kfold / 3class

- Data: walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | nrem sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|
| tcn_hr | 76.2% | 0.528 | 0.672 | 56.7% | 78.7% | 76.3% | 0.513 ± 0.143 |
| tcn_motion_hr | 75.9% | 0.526 | 0.681 | 62.5% | 77.3% | 76.7% | 0.516 ± 0.139 |
| gru_hr | 71.6% | 0.473 | 0.638 | 62.5% | 70.9% | 77.5% | 0.469 ± 0.168 |
| gru_motion_hr | 73.9% | 0.490 | 0.650 | 68.2% | 77.2% | 65.7% | 0.471 ± 0.154 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| tcn_hr | -7.6 [-53.5, +38.3] | +8.6 [-30.4, +47.6] | -1.8% [-13.5%, +10.0%] | +0.9 [-17.1, +18.9] |
| tcn_motion_hr | -9.5 [-50.4, +31.4] | +9.7 [-23.6, +43.1] | -2.4% [-13.0%, +8.2%] | +1.2 [-21.2, +23.6] |
| gru_hr | -20.0 [-71.5, +31.5] | +10.8 [-28.4, +50.1] | -4.4% [-15.9%, +7.1%] | +11.1 [-20.2, +42.4] |
| gru_motion_hr | -24.0 [-106.2, +58.3] | +22.6 [-52.9, +98.1] | -5.8% [-24.0%, +12.5%] | +1.6 [-20.7, +24.0] |
