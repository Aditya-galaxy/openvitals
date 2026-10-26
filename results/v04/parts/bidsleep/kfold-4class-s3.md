# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 47 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.332 [0.297, 0.363] | – | 52.1% [49.5%, 54.4%] | 0.525 | 54.4% | 41.7% | 64.0% | 61.1% | 0.329 ± 0.116 |
| gbdt_motion_hr | 1 | 0.347 [0.311, 0.380] | – | 54.2% [51.7%, 56.6%] | 0.545 | 57.6% | 49.9% | 60.2% | 56.2% | 0.343 ± 0.120 |
| tcn_hr | 1 | 0.490 [0.435, 0.537] | – | 64.0% [60.1%, 67.3%] | 0.635 | 58.5% | 55.2% | 73.9% | 74.6% | 0.488 ± 0.176 |
| tcn_motion_hr | 1 | 0.474 [0.421, 0.519] | – | 63.1% [59.4%, 66.3%] | 0.627 | 65.8% | 57.1% | 69.7% | 67.6% | 0.470 ± 0.180 |
| gru_hr | 1 | 0.453 [0.400, 0.502] | – | 61.1% [57.2%, 64.8%] | 0.608 | 59.2% | 52.1% | 73.2% | 69.2% | 0.454 ± 0.173 |
| gru_motion_hr | 1 | 0.460 [0.409, 0.506] | – | 61.8% [58.1%, 65.2%] | 0.614 | 62.4% | 55.4% | 76.6% | 62.3% | 0.462 ± 0.173 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | -0.016 | [-0.042, +0.009] | 0.893 |
| gru_motion_hr | -0.030 | [-0.053, -0.008] | 0.997 |
| gru_hr | -0.037 | [-0.054, -0.020] | 1.000 |
| gbdt_motion_hr | -0.143 | [-0.176, -0.111] | 1.000 |
| gbdt_hr | -0.158 | [-0.185, -0.131] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.8 [-52.3, +44.8] | +6.7 [-29.5, +42.8] | -1.1% [-12.6%, +10.3%] | -2.4 [-34.0, +29.1] |
| gbdt_motion_hr | -5.1 [-53.2, +43.1] | +9.1 [-32.5, +50.7] | -1.3% [-12.1%, +9.6%] | -3.9 [-34.0, +26.2] |
| tcn_hr | -0.4 [-47.8, +47.0] | -0.2 [-43.2, +42.7] | -0.2% [-10.7%, +10.2%] | +0.4 [-22.5, +23.3] |
| tcn_motion_hr | -12.9 [-72.8, +47.0] | +11.3 [-43.4, +66.0] | -2.9% [-16.1%, +10.4%] | +0.7 [-28.3, +29.7] |
| gru_hr | -4.1 [-56.4, +48.1] | +0.9 [-44.2, +45.9] | -1.1% [-12.7%, +10.5%] | +2.6 [-30.4, +35.6] |
| gru_motion_hr | -7.2 [-80.1, +65.6] | +5.6 [-59.2, +70.4] | -1.5% [-16.6%, +13.7%] | +0.5 [-27.4, +28.4] |
