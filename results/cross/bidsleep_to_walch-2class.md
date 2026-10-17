# OpenVitals benchmark: bidsleep_to_walch / 2class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | sleep sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|
| majority | 91.4% | 0.000 | 0.477 | 0.0% | 100.0% | 0.000 ± 0.000 |
| logreg_hr | 75.5% | 0.206 | 0.580 | 63.4% | 76.7% | 0.206 ± 0.140 |
| gbdt_hr | 81.7% | 0.283 | 0.632 | 62.1% | 83.6% | 0.293 ± 0.157 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| majority | +36.2 [+1.5, +70.9] | -20.3 [-50.2, +9.7] | +9.4% [-1.2%, +19.9%] | -13.0 [-30.6, +4.7] |
| logreg_hr | -76.0 [-146.3, -5.6] | +69.8 [+3.9, +135.6] | -19.2% [-39.0%, +0.6%] | +9.0 [-43.0, +60.9] |
| gbdt_hr | -49.1 [-120.0, +21.9] | +43.9 [-20.4, +108.2] | -12.2% [-30.4%, +6.0%] | +7.3 [-16.1, +30.7] |
