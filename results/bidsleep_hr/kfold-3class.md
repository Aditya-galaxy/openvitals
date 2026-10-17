# OpenVitals benchmark: kfold / 3class

- Data: bidsleep (47 subjects, 253 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | nrem sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|
| majority | 63.1% | 0.000 | 0.258 | 0.0% | 100.0% | 0.0% | 0.000 ± 0.000 |
| logreg_hr | 55.9% | 0.277 | 0.508 | 65.6% | 56.6% | 50.2% | 0.277 ± 0.075 |
| gbdt_hr | 62.5% | 0.368 | 0.587 | 59.0% | 59.5% | 71.1% | 0.371 ± 0.081 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| majority | +44.7 [-12.7, +102.2] | -17.9 [-50.9, +15.1] | +10.8% [-2.8%, +24.4%] | -24.6 [-69.3, +20.1] |
| logreg_hr | -59.6 [-134.8, +15.7] | +58.7 [-1.9, +119.3] | -14.9% [-34.5%, +4.7%] | +2.9 [-56.8, +62.6] |
| gbdt_hr | -13.8 [-75.2, +47.6] | +14.2 [-25.0, +53.5] | -3.6% [-18.7%, +11.6%] | +0.9 [-38.8, +40.6] |
