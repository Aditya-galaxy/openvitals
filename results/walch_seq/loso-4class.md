# OpenVitals benchmark: loso / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 31 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|
| tcn_hr | 67.6% | 0.511 | 0.646 | 60.7% | 62.8% | 76.3% | 77.0% | 0.501 ± 0.124 |
| tcn_motion_hr | 66.9% | 0.500 | 0.646 | 66.8% | 62.9% | 74.9% | 72.2% | 0.482 ± 0.142 |
| gru_hr | 61.3% | 0.432 | 0.580 | 58.7% | 56.0% | 70.2% | 70.4% | 0.415 ± 0.139 |
| gru_motion_hr | 63.0% | 0.452 | 0.607 | 69.0% | 58.0% | 72.0% | 67.9% | 0.434 ± 0.134 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| tcn_hr | -9.2 [-53.0, +34.7] | +9.7 [-30.8, +50.2] | -1.9% [-12.2%, +8.3%] | +2.2 [-16.2, +20.7] |
| tcn_motion_hr | -9.5 [-47.9, +28.9] | +7.4 [-27.1, +41.8] | -2.9% [-14.3%, +8.5%] | +2.9 [-19.4, +25.3] |
| gru_hr | -19.5 [-71.9, +32.8] | +17.9 [-29.4, +65.2] | -5.0% [-17.9%, +7.9%] | +3.7 [-20.6, +28.0] |
| gru_motion_hr | -20.3 [-77.0, +36.4] | +18.0 [-40.4, +76.4] | -4.6% [-18.5%, +9.2%] | +3.7 [-19.1, +26.5] |
