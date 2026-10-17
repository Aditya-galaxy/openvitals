# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 253 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|
| tcn_hr | 65.3% | 0.507 | 0.644 | 61.6% | 60.7% | 73.2% | 68.8% | 0.503 ± 0.096 |
| gru_hr | 66.8% | 0.528 | 0.661 | 62.4% | 61.8% | 74.5% | 71.6% | 0.524 ± 0.110 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| tcn_hr | -5.6 [-67.8, +56.5] | +5.0 [-36.6, +46.6] | -1.5% [-16.5%, +13.5%] | +0.2 [-38.8, +39.2] |
| gru_hr | -4.0 [-62.8, +54.8] | +0.9 [-37.1, +38.9] | -1.1% [-15.2%, +13.0%] | +3.2 [-37.0, +43.4] |
