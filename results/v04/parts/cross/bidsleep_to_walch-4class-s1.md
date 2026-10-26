# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 47 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.293 [0.263, 0.322] | – | 49.2% [47.1%, 51.3%] | 0.481 | 45.7% | 37.0% | 65.1% | 71.9% | 0.285 ± 0.078 |
| gbdt_motion_hr | 1 | 0.322 [0.292, 0.349] | – | 53.2% [51.0%, 55.2%] | 0.516 | 46.0% | 45.3% | 64.9% | 68.6% | 0.311 ± 0.083 |
| tcn_hr | 1 | 0.455 [0.407, 0.499] | – | 62.5% [59.4%, 65.4%] | 0.596 | 48.3% | 53.7% | 80.0% | 79.7% | 0.442 ± 0.134 |
| tcn_motion_hr | 1 | 0.451 [0.411, 0.490] | – | 61.8% [58.9%, 64.5%] | 0.593 | 46.9% | 51.7% | 82.7% | 80.3% | 0.434 ± 0.118 |
| gru_hr | 1 | 0.433 [0.386, 0.476] | – | 60.2% [56.9%, 63.1%] | 0.579 | 54.8% | 49.1% | 78.8% | 78.9% | 0.418 ± 0.140 |
| gru_motion_hr | 1 | 0.425 [0.380, 0.467] | – | 58.9% [55.6%, 61.9%] | 0.580 | 55.7% | 44.2% | 81.4% | 83.3% | 0.410 ± 0.123 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | -0.004 | [-0.023, +0.014] | 0.668 |
| gru_hr | -0.022 | [-0.041, -0.002] | 0.983 |
| gru_motion_hr | -0.031 | [-0.053, -0.009] | 0.996 |
| gbdt_motion_hr | -0.133 | [-0.158, -0.107] | 1.000 |
| gbdt_hr | -0.162 | [-0.187, -0.136] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -2.1 [-48.3, +44.1] | +4.1 [-33.9, +42.0] | -0.9% [-13.5%, +11.7%] | +0.9 [-20.3, +22.0] |
| gbdt_motion_hr | +3.7 [-31.3, +38.7] | +0.0 [-30.7, +30.7] | +0.8% [-9.5%, +11.1%] | -1.0 [-21.4, +19.4] |
| tcn_hr | +2.0 [-41.0, +45.0] | -3.3 [-34.2, +27.6] | +0.1% [-11.5%, +11.8%] | +3.4 [-18.2, +25.0] |
| tcn_motion_hr | +5.6 [-34.5, +45.7] | -2.4 [-33.6, +28.8] | +1.0% [-10.0%, +12.1%] | -0.9 [-20.4, +18.6] |
| gru_hr | -10.0 [-58.9, +39.0] | +5.9 [-32.5, +44.2] | -3.0% [-17.2%, +11.2%] | +5.5 [-24.4, +35.4] |
| gru_motion_hr | -1.1 [-40.6, +38.4] | -2.5 [-35.4, +30.5] | -0.7% [-12.7%, +11.2%] | +5.7 [-11.9, +23.3] |
