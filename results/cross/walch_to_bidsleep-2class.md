# OpenVitals benchmark: walch_to_bidsleep / 2class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 253 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | sleep sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|
| majority | 89.2% | 0.000 | 0.472 | 0.0% | 100.0% | 0.000 ± 0.000 |
| logreg_hr | 74.4% | 0.254 | 0.606 | 70.3% | 74.9% | 0.256 ± 0.087 |
| gbdt_hr | 83.5% | 0.351 | 0.671 | 60.0% | 86.4% | 0.349 ± 0.090 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| majority | +44.7 [-12.7, +102.2] | -17.9 [-50.9, +15.1] | +10.8% [-2.8%, +24.4%] | -24.6 [-69.3, +20.1] |
| logreg_hr | -79.7 [-175.2, +15.8] | +89.8 [+15.5, +164.2] | -19.8% [-43.8%, +4.3%] | -9.2 [-56.0, +37.7] |
| gbdt_hr | -32.6 [-100.3, +35.0] | +33.3 [-12.7, +79.4] | -8.0% [-24.5%, +8.4%] | +0.3 [-41.4, +42.1] |
