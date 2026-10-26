# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 253 nights), walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.360 [0.339, 0.379] | – | 53.6% [52.0%, 55.0%] | 0.537 | 54.6% | 40.5% | 68.1% | 66.0% | 0.341 ± 0.086 |
| tcn_hr | 1 | 0.510 [0.483, 0.533] | – | 65.5% [63.6%, 67.1%] | 0.646 | 58.6% | 58.7% | 78.2% | 71.3% | 0.496 ± 0.122 |
| gru_hr | 1 | 0.526 [0.499, 0.548] | – | 66.6% [64.8%, 68.2%] | 0.656 | 64.4% | 60.8% | 73.9% | 72.6% | 0.503 ± 0.123 |

## Paired comparison against `gbdt_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | +0.165 | [+0.152, +0.178] | 0.000 |
| tcn_hr | +0.150 | [+0.136, +0.163] | 0.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.8 [-61.0, +53.4] | +6.0 [-30.3, +42.3] | -1.1% [-15.1%, +12.9%] | -0.7 [-41.2, +39.9] |
| tcn_hr | -0.9 [-57.9, +56.1] | +1.0 [-38.0, +40.0] | -0.3% [-13.8%, +13.2%] | +0.0 [-38.9, +39.0] |
| gru_hr | -9.2 [-66.2, +47.8] | +3.6 [-36.3, +43.5] | -2.4% [-16.3%, +11.6%] | +5.1 [-35.2, +45.5] |
