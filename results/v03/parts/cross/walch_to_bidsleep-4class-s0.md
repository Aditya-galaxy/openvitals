# OpenVitals benchmark: walch_to_bidsleep / 4class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 253 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.302 [0.281, 0.322] | – | 51.0% [49.6%, 52.4%] | 0.497 | 53.3% | 52.9% | 52.8% | 45.7% | 0.297 ± 0.074 |
| tcn_hr | 1 | 0.426 [0.401, 0.451] | – | 59.0% [57.2%, 60.7%] | 0.575 | 51.8% | 52.5% | 74.1% | 62.0% | 0.423 ± 0.091 |
| gru_hr | 1 | 0.399 [0.370, 0.426] | – | 57.9% [55.8%, 59.9%] | 0.563 | 59.7% | 60.0% | 61.3% | 51.1% | 0.394 ± 0.100 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | -0.028 | [-0.042, -0.015] | 1.000 |
| gbdt_hr | -0.124 | [-0.140, -0.110] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -12.6 [-73.3, +48.2] | +15.7 [-24.0, +55.5] | -3.2% [-18.0%, +11.6%] | -1.4 [-44.1, +41.3] |
| tcn_hr | -10.7 [-73.5, +52.1] | +19.1 [-23.3, +61.4] | -2.7% [-17.6%, +12.2%] | -7.3 [-60.0, +45.5] |
| gru_hr | -18.5 [-79.6, +42.5] | +21.6 [-26.3, +69.4] | -4.6% [-19.2%, +10.0%] | -2.6 [-56.3, +51.1] |
