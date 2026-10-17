# OpenVitals benchmark: walch_to_bidsleep / 4class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 253 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|
| majority | 44.2% | 0.000 | 0.153 | 0.0% | 100.0% | 0.0% | 0.0% | 0.000 ± 0.000 |
| logreg_hr | 45.3% | 0.260 | 0.432 | 45.6% | 41.5% | 71.4% | 32.7% | 0.256 ± 0.071 |
| gbdt_hr | 51.0% | 0.302 | 0.497 | 53.3% | 52.9% | 52.8% | 45.7% | 0.297 ± 0.074 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| majority | +44.7 [-12.7, +102.2] | -17.9 [-50.9, +15.1] | +10.8% [-2.8%, +24.4%] | -24.6 [-69.3, +20.1] |
| logreg_hr | -40.3 [-127.7, +47.1] | +63.6 [-3.9, +131.1] | -10.2% [-32.1%, +11.7%] | -21.2 [-67.8, +25.4] |
| gbdt_hr | -12.6 [-73.3, +48.2] | +15.7 [-24.0, +55.5] | -3.2% [-18.0%, +11.6%] | -1.4 [-44.1, +41.3] |
