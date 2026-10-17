# OpenVitals benchmark: walch_to_bidsleep / 4class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 253 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|
| tcn_hr | 59.0% | 0.426 | 0.575 | 51.8% | 52.5% | 74.1% | 62.0% | 0.423 ± 0.091 |
| gru_hr | 57.9% | 0.399 | 0.563 | 59.7% | 60.0% | 61.3% | 51.1% | 0.394 ± 0.100 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| tcn_hr | -10.7 [-73.5, +52.1] | +19.1 [-23.3, +61.4] | -2.7% [-17.6%, +12.2%] | -7.3 [-60.0, +45.5] |
| gru_hr | -18.5 [-79.6, +42.5] | +21.6 [-26.3, +69.4] | -4.6% [-19.2%, +10.0%] | -2.6 [-56.3, +51.1] |
