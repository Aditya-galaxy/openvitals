# OpenVitals benchmark: kfold / 2class

- Data: walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | sleep sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|
| tcn_hr | 90.1% | 0.437 | 0.718 | 55.1% | 93.4% | 0.434 ± 0.170 |
| tcn_motion_hr | 91.0% | 0.487 | 0.743 | 59.8% | 94.0% | 0.502 ± 0.178 |
| gru_hr | 88.9% | 0.425 | 0.711 | 60.2% | 91.6% | 0.438 ± 0.203 |
| gru_motion_hr | 89.4% | 0.468 | 0.732 | 67.5% | 91.5% | 0.478 ± 0.184 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| tcn_hr | -8.9 [-62.1, +44.3] | +8.6 [-38.2, +55.5] | -2.2% [-15.9%, +11.6%] | +2.1 [-14.7, +18.9] |
| tcn_motion_hr | -8.4 [-73.6, +56.8] | +9.9 [-48.1, +68.0] | -1.7% [-16.6%, +13.3%] | +0.6 [-22.1, +23.4] |
| gru_hr | -17.6 [-78.9, +43.7] | +16.0 [-38.0, +70.0] | -4.0% [-18.5%, +10.5%] | +3.5 [-19.1, +26.2] |
| gru_motion_hr | -20.9 [-84.6, +42.8] | +19.5 [-40.2, +79.2] | -5.0% [-20.0%, +9.9%] | +1.3 [-23.5, +26.0] |
