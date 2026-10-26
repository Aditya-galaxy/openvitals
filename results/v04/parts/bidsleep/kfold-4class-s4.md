# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 47 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.337 [0.302, 0.368] | – | 52.6% [50.0%, 54.9%] | 0.528 | 54.4% | 42.5% | 63.9% | 61.5% | 0.334 ± 0.118 |
| gbdt_motion_hr | 1 | 0.349 [0.313, 0.383] | – | 54.4% [51.8%, 56.8%] | 0.547 | 58.5% | 50.1% | 59.3% | 56.8% | 0.345 ± 0.122 |
| tcn_hr | 1 | 0.491 [0.439, 0.535] | – | 64.3% [60.6%, 67.4%] | 0.639 | 58.1% | 57.8% | 75.8% | 69.8% | 0.488 ± 0.171 |
| tcn_motion_hr | 1 | 0.457 [0.407, 0.503] | – | 61.9% [58.3%, 65.2%] | 0.611 | 58.5% | 55.9% | 69.7% | 67.9% | 0.458 ± 0.172 |
| gru_hr | 1 | 0.454 [0.396, 0.506] | – | 61.1% [56.8%, 64.9%] | 0.609 | 59.9% | 51.2% | 72.8% | 70.7% | 0.456 ± 0.179 |
| gru_motion_hr | 1 | 0.435 [0.389, 0.476] | – | 59.7% [56.2%, 62.7%] | 0.599 | 62.7% | 49.2% | 73.9% | 66.9% | 0.438 ± 0.155 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | -0.034 | [-0.055, -0.012] | 0.999 |
| gru_hr | -0.037 | [-0.060, -0.012] | 0.998 |
| gru_motion_hr | -0.055 | [-0.080, -0.030] | 1.000 |
| gbdt_motion_hr | -0.142 | [-0.177, -0.110] | 1.000 |
| gbdt_hr | -0.153 | [-0.179, -0.129] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.6 [-51.1, +43.8] | +7.1 [-28.2, +42.4] | -1.1% [-12.2%, +9.9%] | -3.1 [-35.0, +28.7] |
| gbdt_motion_hr | -5.3 [-53.8, +43.2] | +7.6 [-37.5, +52.7] | -1.4% [-12.5%, +9.7%] | -2.1 [-34.7, +30.4] |
| tcn_hr | +1.9 [-38.5, +42.3] | -4.9 [-40.1, +30.2] | +0.4% [-8.5%, +9.3%] | +3.0 [-22.7, +28.7] |
| tcn_motion_hr | -3.7 [-69.5, +62.1] | +5.2 [-49.8, +60.1] | -0.7% [-15.0%, +13.5%] | -2.3 [-37.5, +33.0] |
| gru_hr | -5.7 [-62.6, +51.2] | +2.0 [-45.1, +49.1] | -1.6% [-14.7%, +11.6%] | +2.3 [-28.8, +33.4] |
| gru_motion_hr | -10.4 [-77.9, +57.1] | +6.7 [-51.4, +64.7] | -2.4% [-17.3%, +12.5%] | +3.2 [-34.7, +41.0] |
