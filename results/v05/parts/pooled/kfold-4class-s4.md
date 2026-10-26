# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 253 nights), walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.362 [0.340, 0.380] | – | 53.7% [52.1%, 55.1%] | 0.538 | 54.6% | 40.7% | 68.1% | 66.0% | 0.342 ± 0.085 |
| tcn_hr | 1 | 0.507 [0.479, 0.531] | – | 65.5% [63.6%, 67.2%] | 0.644 | 54.0% | 61.0% | 79.2% | 68.6% | 0.494 ± 0.129 |
| gru_hr | 1 | 0.532 [0.505, 0.555] | – | 67.2% [65.5%, 68.9%] | 0.661 | 59.9% | 62.9% | 78.9% | 69.7% | 0.509 ± 0.124 |

## Paired comparison against `gbdt_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | +0.170 | [+0.156, +0.185] | 0.000 |
| tcn_hr | +0.146 | [+0.131, +0.162] | 0.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.6 [-60.9, +53.6] | +6.0 [-30.2, +42.1] | -1.1% [-15.0%, +12.9%] | -0.8 [-40.9, +39.3] |
| tcn_hr | +6.0 [-49.0, +61.0] | -4.2 [-40.2, +31.7] | +1.3% [-11.9%, +14.6%] | -1.4 [-44.1, +41.3] |
| gru_hr | -1.5 [-55.8, +52.8] | +0.3 [-38.3, +39.0] | -0.5% [-13.9%, +12.8%] | +0.6 [-36.6, +37.8] |
