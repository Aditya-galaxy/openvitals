# OpenVitals benchmark: kfold / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.312 [0.272, 0.351] | – | 53.6% [51.2%, 56.0%] | 0.497 | 49.1% | 51.8% | 53.0% | 60.4% | 0.309 ± 0.111 |
| gbdt_motion_hr | 1 | 0.371 [0.331, 0.411] | – | 60.3% [57.7%, 62.7%] | 0.552 | 59.0% | 65.2% | 45.3% | 57.6% | 0.360 ± 0.120 |
| tcn_hr | 1 | 0.468 [0.417, 0.513] | – | 64.9% [61.7%, 67.8%] | 0.612 | 55.1% | 63.0% | 75.4% | 67.0% | 0.454 ± 0.141 |
| tcn_motion_hr | 1 | 0.468 [0.425, 0.510] | – | 65.5% [62.7%, 68.3%] | 0.614 | 64.1% | 66.1% | 63.4% | 66.0% | 0.456 ± 0.123 |
| gru_hr | 1 | 0.433 [0.377, 0.482] | – | 61.2% [57.5%, 64.6%] | 0.580 | 57.9% | 55.1% | 73.8% | 70.1% | 0.423 ± 0.141 |
| gru_motion_hr | 1 | 0.450 [0.400, 0.494] | – | 63.4% [60.2%, 66.3%] | 0.601 | 65.6% | 61.0% | 71.7% | 63.4% | 0.432 ± 0.140 |

## Paired comparison against `tcn_motion_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.001 | [-0.034, +0.034] | 0.505 |
| gru_motion_hr | -0.018 | [-0.053, +0.019] | 0.830 |
| gru_hr | -0.035 | [-0.075, +0.003] | 0.960 |
| gbdt_motion_hr | -0.097 | [-0.137, -0.054] | 1.000 |
| gbdt_hr | -0.156 | [-0.193, -0.116] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -11.3 [-64.3, +41.7] | +12.0 [-31.3, +55.3] | -2.6% [-15.8%, +10.5%] | +2.1 [-17.8, +22.1] |
| gbdt_motion_hr | -9.3 [-55.0, +36.4] | +10.3 [-28.4, +49.0] | -2.3% [-13.8%, +9.2%] | +1.4 [-19.8, +22.7] |
| tcn_hr | -5.0 [-51.5, +41.4] | +6.5 [-35.2, +48.3] | -1.7% [-14.7%, +11.3%] | +0.6 [-15.2, +16.3] |
| tcn_motion_hr | -16.2 [-87.8, +55.4] | +17.4 [-47.5, +82.2] | -3.8% [-19.6%, +11.9%] | +0.3 [-20.6, +21.3] |
| gru_hr | -18.3 [-71.5, +34.9] | +14.0 [-33.1, +61.1] | -4.6% [-18.1%, +8.8%] | +5.4 [-21.3, +32.1] |
| gru_motion_hr | -18.3 [-79.7, +43.0] | +18.2 [-33.0, +69.4] | -4.3% [-21.2%, +12.7%] | +1.3 [-24.2, +26.8] |
