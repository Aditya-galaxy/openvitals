# OpenVitals benchmark: loso / 2class

- Data: walch (31 subjects, 31 nights)
- Folds: 31 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | sleep sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|
| tcn_hr | 91.5% | 0.501 | 0.751 | 59.3% | 94.6% | 0.485 ± 0.189 |
| tcn_motion_hr | 92.3% | 0.532 | 0.766 | 60.1% | 95.3% | 0.522 ± 0.161 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| tcn_hr | -5.9 [-47.8, +35.9] | +5.1 [-34.7, +44.8] | -1.3% [-11.1%, +8.5%] | +2.8 [-15.6, +21.1] |
| tcn_motion_hr | -3.4 [-46.4, +39.7] | +4.8 [-31.8, +41.4] | -0.8% [-11.2%, +9.7%] | +0.6 [-21.2, +22.5] |
