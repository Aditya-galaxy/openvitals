# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 253 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.364 [0.341, 0.385] | – | 53.8% [52.1%, 55.3%] | 0.540 | 55.1% | 40.7% | 68.2% | 65.0% | 0.362 ± 0.074 |
| tcn_hr | 1 | 0.510 [0.480, 0.537] | – | 65.3% [63.1%, 67.3%] | 0.644 | 57.7% | 57.9% | 78.0% | 71.7% | 0.505 ± 0.106 |
| gru_hr | 1 | 0.523 [0.493, 0.548] | – | 66.1% [64.1%, 67.9%] | 0.654 | 62.9% | 58.5% | 77.7% | 71.9% | 0.518 ± 0.105 |

## Paired comparison against `gru_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.013 | [-0.022, -0.003] | 0.996 |
| gbdt_hr | -0.159 | [-0.173, -0.144] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.9 [-63.6, +55.9] | +6.7 [-29.9, +43.4] | -1.1% [-15.6%, +13.4%] | -1.6 [-44.4, +41.1] |
| tcn_hr | -0.3 [-58.2, +57.6] | +2.0 [-35.1, +39.1] | -0.2% [-14.0%, +13.6%] | -2.1 [-44.7, +40.6] |
| gru_hr | -5.9 [-63.8, +51.9] | +4.2 [-36.5, +44.9] | -1.5% [-15.1%, +12.1%] | +1.0 [-42.2, +44.2] |
