# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 253 nights), walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 5; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 5 | 0.361 [0.339, 0.380] | 0.000 | 53.6% [52.0%, 55.0%] | 0.538 | 54.5% | 40.6% | 68.1% | 65.9% | 0.341 ± 0.085 |
| tcn_hr | 5 | 0.509 [0.481, 0.532] | 0.003 | 65.5% [63.6%, 67.1%] | 0.645 | 56.7% | 59.0% | 78.4% | 71.4% | 0.493 ± 0.118 |
| gru_hr | 5 | 0.527 [0.501, 0.549] | 0.003 | 66.8% [65.1%, 68.4%] | 0.658 | 61.2% | 61.7% | 77.2% | 70.9% | 0.506 ± 0.116 |

## Paired comparison against `gru_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.018 | [-0.022, -0.014] | 1.000 |
| gbdt_hr | -0.167 | [-0.179, -0.154] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.6 [-60.8, +53.6] | +5.9 [-30.0, +41.8] | -1.1% [-15.0%, +12.9%] | -0.7 [-40.7, +39.2] |
| tcn_hr | +1.9 [-53.0, +56.7] | -1.4 [-38.1, +35.3] | +0.3% [-12.8%, +13.5%] | -0.2 [-41.2, +40.8] |
| gru_hr | -3.4 [-57.9, +51.0] | +1.2 [-36.9, +39.3] | -1.0% [-14.3%, +12.4%] | +1.7 [-37.0, +40.3] |
