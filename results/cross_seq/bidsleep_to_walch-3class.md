# OpenVitals benchmark: bidsleep_to_walch / 3class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | nrem sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|
| tcn_hr | 75.6% | 0.538 | 0.669 | 59.2% | 75.2% | 83.5% | 0.531 ± 0.151 |
| gru_hr | 76.3% | 0.552 | 0.674 | 57.0% | 75.4% | 87.0% | 0.537 ± 0.150 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| tcn_hr | -15.1 [-59.8, +29.6] | +14.5 [-23.6, +52.6] | -3.7% [-13.6%, +6.2%] | +2.0 [-14.2, +18.2] |
| gru_hr | -13.1 [-65.8, +39.6] | +6.7 [-38.9, +52.4] | -3.7% [-17.8%, +10.3%] | +8.2 [-15.0, +31.5] |
