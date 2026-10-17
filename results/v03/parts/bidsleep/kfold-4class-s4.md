# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 253 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.364 [0.341, 0.384] | – | 53.8% [52.1%, 55.3%] | 0.540 | 54.8% | 40.6% | 68.4% | 65.0% | 0.361 ± 0.074 |
| tcn_hr | 1 | 0.517 [0.486, 0.545] | – | 65.9% [63.7%, 67.9%] | 0.650 | 56.4% | 59.3% | 78.8% | 71.6% | 0.513 ± 0.103 |
| gru_hr | 1 | 0.533 [0.504, 0.559] | – | 66.9% [64.9%, 68.9%] | 0.663 | 66.1% | 61.0% | 77.0% | 70.1% | 0.525 ± 0.102 |

## Paired comparison against `gru_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.016 | [-0.028, -0.004] | 0.993 |
| gbdt_hr | -0.169 | [-0.186, -0.151] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.6 [-63.1, +56.0] | +6.6 [-29.7, +42.9] | -1.1% [-15.5%, +13.4%] | -1.8 [-44.5, +41.0] |
| tcn_hr | +2.8 [-52.7, +58.3] | -0.9 [-35.5, +33.6] | +0.6% [-12.8%, +13.9%] | -1.5 [-43.0, +40.0] |
| gru_hr | -8.4 [-63.7, +47.0] | +3.6 [-34.3, +41.5] | -2.1% [-15.6%, +11.3%] | +4.2 [-33.2, +41.5] |
