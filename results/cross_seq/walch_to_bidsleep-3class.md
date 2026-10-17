# OpenVitals benchmark: walch_to_bidsleep / 3class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 253 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | nrem sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|
| tcn_hr | 72.7% | 0.453 | 0.627 | 44.0% | 85.2% | 54.5% | 0.452 ± 0.115 |
| gru_hr | 69.0% | 0.413 | 0.613 | 58.5% | 77.9% | 51.9% | 0.411 ± 0.107 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| tcn_hr | +5.4 [-60.2, +71.0] | +5.1 [-33.3, +43.5] | +1.3% [-14.1%, +16.6%] | -10.3 [-60.8, +40.3] |
| gru_hr | -14.0 [-75.7, +47.6] | +17.9 [-25.6, +61.4] | -3.5% [-18.2%, +11.2%] | -3.4 [-51.1, +44.3] |
