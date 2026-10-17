# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|
| tcn_hr | 61.4% | 0.455 | 0.590 | 47.6% | 47.4% | 81.9% | 89.5% | 0.446 ± 0.132 |
| gru_hr | 63.5% | 0.483 | 0.615 | 60.2% | 49.8% | 83.1% | 87.2% | 0.467 ± 0.135 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| tcn_hr | +2.4 [-34.0, +38.9] | -4.0 [-33.1, +25.2] | +0.3% [-8.2%, +8.8%] | +4.2 [-12.3, +20.8] |
| gru_hr | -12.2 [-56.3, +31.8] | +4.2 [-25.5, +33.8] | -3.3% [-15.2%, +8.6%] | +7.8 [-16.3, +31.9] |
