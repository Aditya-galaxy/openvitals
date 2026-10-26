# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 253 nights), walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.361 [0.339, 0.380] | – | 53.6% [52.0%, 55.0%] | 0.538 | 54.6% | 40.6% | 68.2% | 65.8% | 0.341 ± 0.086 |
| tcn_hr | 1 | 0.514 [0.486, 0.538] | – | 65.8% [63.8%, 67.4%] | 0.650 | 57.7% | 58.3% | 78.5% | 73.1% | 0.496 ± 0.113 |
| gru_hr | 1 | 0.527 [0.500, 0.551] | – | 66.8% [64.9%, 68.4%] | 0.658 | 61.3% | 60.9% | 78.5% | 71.1% | 0.505 ± 0.116 |

## Paired comparison against `gbdt_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | +0.167 | [+0.152, +0.181] | 0.000 |
| tcn_hr | +0.154 | [+0.139, +0.167] | 0.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.6 [-60.5, +53.4] | +5.9 [-29.9, +41.7] | -1.1% [-15.0%, +12.8%] | -0.8 [-40.9, +39.3] |
| tcn_hr | +2.1 [-51.3, +55.5] | -2.2 [-37.8, +33.4] | +0.4% [-12.5%, +13.3%] | +0.5 [-39.7, +40.8] |
| gru_hr | -2.8 [-55.3, +49.6] | +0.4 [-35.0, +35.8] | -0.8% [-13.9%, +12.2%] | +2.3 [-38.6, +43.2] |
