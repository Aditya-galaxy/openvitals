# OpenVitals benchmark: bidsleep_to_walch / 2class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | sleep sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|
| tcn_hr | 86.5% | 0.354 | 0.674 | 57.5% | 89.2% | 0.355 ± 0.194 |
| gru_hr | 88.5% | 0.409 | 0.703 | 59.1% | 91.3% | 0.402 ± 0.219 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| tcn_hr | -25.8 [-85.0, +33.4] | +17.6 [-33.8, +69.0] | -6.6% [-22.8%, +9.6%] | +10.9 [-10.1, +32.0] |
| gru_hr | -18.5 [-68.6, +31.5] | +7.3 [-39.1, +53.6] | -5.1% [-19.3%, +9.2%] | +12.3 [-9.7, +34.3] |
