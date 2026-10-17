# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 253 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.364 [0.341, 0.385] | – | 53.8% [52.1%, 55.4%] | 0.540 | 55.1% | 40.8% | 68.1% | 64.9% | 0.362 ± 0.074 |
| tcn_hr | 1 | 0.507 [0.478, 0.532] | – | 65.3% [63.2%, 67.1%] | 0.644 | 61.6% | 60.7% | 73.2% | 68.8% | 0.503 ± 0.096 |
| gru_hr | 1 | 0.528 [0.496, 0.557] | – | 66.8% [64.5%, 68.9%] | 0.661 | 62.4% | 61.8% | 74.5% | 71.6% | 0.524 ± 0.110 |

## Paired comparison against `gru_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.021 | [-0.033, -0.010] | 1.000 |
| gbdt_hr | -0.164 | [-0.181, -0.147] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -4.1 [-63.7, +55.5] | +7.1 [-29.6, +43.8] | -1.2% [-15.7%, +13.3%] | -1.7 [-43.3, +39.8] |
| tcn_hr | -5.6 [-67.8, +56.5] | +5.0 [-36.6, +46.6] | -1.5% [-16.5%, +13.5%] | +0.2 [-38.8, +39.2] |
| gru_hr | -4.0 [-62.8, +54.8] | +0.9 [-37.1, +38.9] | -1.1% [-15.2%, +13.0%] | +3.2 [-37.0, +43.4] |
