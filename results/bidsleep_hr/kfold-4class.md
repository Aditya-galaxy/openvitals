# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 253 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|
| majority | 44.2% | 0.000 | 0.153 | 0.0% | 100.0% | 0.0% | 0.0% | 0.000 ± 0.000 |
| logreg_hr | 47.8% | 0.290 | 0.463 | 47.3% | 41.4% | 70.3% | 42.6% | 0.286 ± 0.068 |
| gbdt_hr | 53.8% | 0.364 | 0.540 | 55.1% | 40.8% | 68.1% | 64.9% | 0.362 ± 0.074 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| majority | +44.7 [-12.7, +102.2] | -17.9 [-50.9, +15.1] | +10.8% [-2.8%, +24.4%] | -24.6 [-69.3, +20.1] |
| logreg_hr | -27.5 [-99.9, +44.9] | +47.8 [-3.1, +98.8] | -7.0% [-25.2%, +11.2%] | -18.2 [-65.8, +29.4] |
| gbdt_hr | -4.1 [-63.7, +55.5] | +7.1 [-29.6, +43.8] | -1.2% [-15.7%, +13.3%] | -1.7 [-43.3, +39.8] |
