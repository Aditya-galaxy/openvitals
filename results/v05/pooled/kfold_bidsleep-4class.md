# OpenVitals benchmark: kfold_bidsleep / 4class

- Data: bidsleep (47 subjects, 253 nights), walch (31 subjects, 31 nights); scored on bidsleep only
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 5; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 5 | 0.366 [0.343, 0.387] | 0.001 | 54.0% [52.3%, 55.6%] | 0.543 | 55.0% | 41.4% | 68.1% | 64.7% | 0.363 ± 0.075 |
| tcn_hr | 5 | 0.512 [0.484, 0.538] | 0.003 | 65.6% [63.6%, 67.4%] | 0.648 | 57.1% | 59.7% | 77.9% | 70.2% | 0.507 ± 0.101 |
| gru_hr | 5 | 0.531 [0.504, 0.556] | 0.003 | 67.0% [65.1%, 68.8%] | 0.661 | 61.5% | 62.5% | 76.7% | 69.9% | 0.523 ± 0.101 |

## Paired comparison against `gru_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.019 | [-0.023, -0.015] | 1.000 |
| gbdt_hr | -0.165 | [-0.179, -0.151] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.3 [-61.9, +55.3] | +5.9 [-30.0, +41.8] | -1.0% [-15.2%, +13.3%] | -1.2 [-43.0, +40.5] |
| tcn_hr | +2.3 [-53.8, +58.4] | -1.4 [-38.4, +35.6] | +0.5% [-12.8%, +13.8%] | -0.7 [-43.6, +42.1] |
| gru_hr | -2.8 [-57.5, +51.9] | +1.1 [-36.2, +38.3] | -0.7% [-13.7%, +12.2%] | +1.0 [-38.9, +41.0] |
