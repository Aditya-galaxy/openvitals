# OpenVitals benchmark: kfold / 2class

- Data: bidsleep (47 subjects, 253 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | sleep sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|
| majority | 89.2% | 0.000 | 0.472 | 0.0% | 100.0% | 0.000 ± 0.000 |
| logreg_hr | 73.4% | 0.252 | 0.602 | 73.1% | 73.5% | 0.249 ± 0.124 |
| gbdt_hr | 83.4% | 0.373 | 0.681 | 65.7% | 85.6% | 0.369 ± 0.124 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| majority | +44.7 [-12.7, +102.2] | -17.9 [-50.9, +15.1] | +10.8% [-2.8%, +24.4%] | -24.6 [-69.3, +20.1] |
| logreg_hr | -86.5 [-170.8, -2.1] | +66.5 [-10.7, +143.8] | -21.5% [-43.7%, +0.8%] | +21.1 [-53.3, +95.5] |
| gbdt_hr | -38.1 [-110.7, +34.4] | +35.1 [-16.4, +86.5] | -9.4% [-27.0%, +8.2%] | +3.4 [-37.6, +44.4] |
