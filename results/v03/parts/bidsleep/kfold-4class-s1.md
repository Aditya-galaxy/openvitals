# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 253 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.363 [0.341, 0.384] | – | 53.8% [52.0%, 55.3%] | 0.540 | 55.1% | 40.6% | 68.3% | 64.8% | 0.361 ± 0.074 |
| tcn_hr | 1 | 0.517 [0.485, 0.545] | – | 65.9% [63.7%, 67.9%] | 0.651 | 59.3% | 60.1% | 73.9% | 72.9% | 0.512 ± 0.101 |
| gru_hr | 1 | 0.534 [0.502, 0.563] | – | 67.2% [64.9%, 69.3%] | 0.665 | 62.1% | 62.2% | 78.3% | 69.7% | 0.527 ± 0.109 |

## Paired comparison against `gru_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.017 | [-0.029, -0.007] | 1.000 |
| gbdt_hr | -0.171 | [-0.188, -0.153] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.9 [-63.6, +55.8] | +7.1 [-29.8, +43.9] | -1.1% [-15.6%, +13.3%] | -1.9 [-43.9, +40.1] |
| tcn_hr | -1.7 [-57.0, +53.6] | +2.5 [-35.4, +40.4] | -0.5% [-13.9%, +12.9%] | -0.5 [-42.4, +41.4] |
| gru_hr | -1.8 [-56.9, +53.2] | +0.6 [-37.9, +39.2] | -0.6% [-13.6%, +12.4%] | +0.4 [-39.0, +39.8] |
