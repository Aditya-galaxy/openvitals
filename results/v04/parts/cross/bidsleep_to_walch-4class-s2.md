# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 47 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.293 [0.263, 0.321] | – | 49.2% [47.0%, 51.3%] | 0.481 | 44.7% | 36.6% | 66.3% | 72.3% | 0.286 ± 0.077 |
| gbdt_motion_hr | 1 | 0.322 [0.291, 0.351] | – | 53.0% [50.8%, 55.1%] | 0.517 | 46.3% | 44.6% | 65.5% | 69.3% | 0.310 ± 0.090 |
| tcn_hr | 1 | 0.414 [0.372, 0.455] | – | 58.8% [55.7%, 61.5%] | 0.563 | 47.8% | 46.3% | 76.9% | 83.5% | 0.400 ± 0.122 |
| tcn_motion_hr | 1 | 0.431 [0.384, 0.474] | – | 60.9% [57.5%, 63.9%] | 0.579 | 36.9% | 51.7% | 83.2% | 79.6% | 0.412 ± 0.141 |
| gru_hr | 1 | 0.379 [0.332, 0.421] | – | 55.4% [52.0%, 58.5%] | 0.534 | 35.9% | 40.5% | 83.6% | 83.1% | 0.367 ± 0.124 |
| gru_motion_hr | 1 | 0.410 [0.363, 0.452] | – | 58.8% [55.2%, 62.0%] | 0.562 | 37.0% | 47.3% | 80.7% | 82.8% | 0.395 ± 0.138 |

## Paired comparison against `tcn_motion_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.017 | [-0.036, +0.003] | 0.956 |
| gru_motion_hr | -0.022 | [-0.047, +0.006] | 0.941 |
| gru_hr | -0.053 | [-0.078, -0.028] | 1.000 |
| gbdt_motion_hr | -0.109 | [-0.134, -0.083] | 1.000 |
| gbdt_hr | -0.138 | [-0.167, -0.107] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -1.3 [-48.2, +45.6] | +2.9 [-36.8, +42.7] | -0.7% [-13.3%, +11.9%] | +1.2 [-17.7, +20.2] |
| gbdt_motion_hr | +4.4 [-31.1, +39.9] | -0.7 [-32.9, +31.5] | +0.9% [-9.9%, +11.7%] | -0.9 [-22.6, +20.7] |
| tcn_hr | -4.1 [-48.9, +40.7] | +2.8 [-29.7, +35.3] | -1.4% [-13.3%, +10.5%] | +4.1 [-17.5, +25.8] |
| tcn_motion_hr | +17.8 [-16.7, +52.3] | -13.4 [-40.3, +13.5] | +4.3% [-3.5%, +12.2%] | -1.6 [-20.4, +17.2] |
| gru_hr | +14.9 [-24.7, +54.4] | -11.4 [-43.7, +21.0] | +3.4% [-5.7%, +12.5%] | -0.8 [-19.7, +18.1] |
| gru_motion_hr | +16.0 [-19.9, +51.8] | -9.3 [-39.2, +20.6] | +3.8% [-4.3%, +12.0%] | -4.4 [-24.7, +15.9] |
