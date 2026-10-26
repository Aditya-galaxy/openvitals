# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 253 nights), walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.360 [0.339, 0.379] | – | 53.6% [52.0%, 55.0%] | 0.537 | 54.4% | 40.6% | 68.1% | 65.9% | 0.341 ± 0.086 |
| tcn_hr | 1 | 0.508 [0.482, 0.531] | – | 65.4% [63.6%, 67.0%] | 0.643 | 54.4% | 58.4% | 78.3% | 73.1% | 0.492 ± 0.116 |
| gru_hr | 1 | 0.527 [0.500, 0.550] | – | 66.9% [65.0%, 68.5%] | 0.659 | 61.1% | 61.9% | 76.9% | 70.9% | 0.508 ± 0.114 |

## Paired comparison against `gbdt_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | +0.166 | [+0.152, +0.180] | 0.000 |
| tcn_hr | +0.148 | [+0.136, +0.160] | 0.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.6 [-60.9, +53.7] | +5.9 [-30.0, +41.7] | -1.1% [-15.1%, +13.0%] | -0.7 [-40.6, +39.2] |
| tcn_hr | +4.1 [-50.8, +59.0] | -3.3 [-38.3, +31.8] | +1.0% [-12.1%, +14.0%] | -0.8 [-42.8, +41.2] |
| gru_hr | -2.5 [-54.4, +49.4] | +1.5 [-36.5, +39.6] | -0.6% [-12.9%, +11.6%] | +0.0 [-37.1, +37.2] |
