# OpenVitals benchmark: walch_to_bidsleep / 2class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 253 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | sleep sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|
| tcn_hr | 90.1% | 0.401 | 0.700 | 38.3% | 96.3% | 0.415 ± 0.121 |
| gru_hr | 87.9% | 0.447 | 0.723 | 59.5% | 91.4% | 0.430 ± 0.108 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| tcn_hr | +14.1 [-48.0, +76.1] | -3.0 [-37.9, +31.8] | +3.4% [-11.3%, +18.0%] | -10.2 [-60.5, +40.1] |
| gru_hr | -13.9 [-70.3, +42.5] | +15.8 [-29.2, +60.7] | -3.4% [-16.9%, +10.2%] | -1.8 [-51.8, +48.2] |
