# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 253 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.365 [0.342, 0.386] | – | 53.8% [52.1%, 55.4%] | 0.541 | 55.3% | 40.6% | 68.2% | 65.3% | 0.363 ± 0.074 |
| tcn_hr | 1 | 0.513 [0.484, 0.539] | – | 65.6% [63.6%, 67.4%] | 0.647 | 56.7% | 59.2% | 77.6% | 71.5% | 0.508 ± 0.099 |
| gru_hr | 1 | 0.521 [0.490, 0.549] | – | 66.3% [64.0%, 68.2%] | 0.654 | 63.4% | 61.9% | 76.5% | 67.3% | 0.516 ± 0.105 |

## Paired comparison against `gru_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.008 | [-0.018, +0.002] | 0.951 |
| gbdt_hr | -0.156 | [-0.174, -0.139] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -4.1 [-63.8, +55.5] | +6.9 [-29.5, +43.4] | -1.2% [-15.7%, +13.3%] | -1.5 [-43.8, +40.7] |
| tcn_hr | +1.3 [-56.2, +58.7] | +1.5 [-36.4, +39.3] | +0.3% [-13.5%, +14.0%] | -2.6 [-44.9, +39.7] |
| gru_hr | -7.5 [-67.4, +52.5] | +4.9 [-35.4, +45.2] | -1.8% [-15.9%, +12.2%] | +1.0 [-38.9, +41.0] |
