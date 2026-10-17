# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 253 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 5; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 5 | 0.364 [0.341, 0.385] | 0.001 | 53.8% [52.1%, 55.3%] | 0.540 | 55.1% | 40.7% | 68.3% | 65.0% | 0.362 ± 0.074 |
| tcn_hr | 5 | 0.513 [0.483, 0.540] | 0.004 | 65.6% [63.5%, 67.5%] | 0.647 | 58.3% | 59.4% | 76.3% | 71.3% | 0.508 ± 0.099 |
| gru_hr | 5 | 0.528 [0.498, 0.554] | 0.006 | 66.7% [64.5%, 68.5%] | 0.660 | 63.4% | 61.1% | 76.8% | 70.1% | 0.522 ± 0.103 |

## Paired comparison against `gbdt_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | +0.164 | [+0.149, +0.179] | 0.000 |
| tcn_hr | +0.149 | [+0.135, +0.162] | 0.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.9 [-63.5, +55.7] | +6.9 [-29.7, +43.5] | -1.1% [-15.6%, +13.4%] | -1.7 [-44.0, +40.6] |
| tcn_hr | -0.7 [-58.4, +57.0] | +2.0 [-35.8, +39.8] | -0.3% [-14.1%, +13.6%] | -1.3 [-42.8, +40.2] |
| gru_hr | -5.5 [-62.9, +51.9] | +2.8 [-36.2, +41.9] | -1.4% [-15.1%, +12.2%] | +2.0 [-38.1, +42.0] |
