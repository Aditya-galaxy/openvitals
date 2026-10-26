# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 47 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.334 [0.300, 0.364] | – | 52.3% [49.8%, 54.6%] | 0.525 | 54.2% | 42.1% | 63.5% | 61.4% | 0.330 ± 0.114 |
| gbdt_motion_hr | 1 | 0.346 [0.312, 0.378] | – | 54.0% [51.5%, 56.3%] | 0.544 | 58.6% | 48.7% | 60.4% | 56.9% | 0.342 ± 0.116 |
| tcn_hr | 1 | 0.460 [0.405, 0.508] | – | 62.2% [58.4%, 65.7%] | 0.611 | 52.3% | 57.9% | 77.4% | 62.9% | 0.457 ± 0.175 |
| tcn_motion_hr | 1 | 0.486 [0.437, 0.531] | – | 64.1% [60.6%, 67.3%] | 0.635 | 61.2% | 59.6% | 75.2% | 65.1% | 0.484 ± 0.171 |
| gru_hr | 1 | 0.448 [0.396, 0.495] | – | 61.0% [57.4%, 64.3%] | 0.605 | 55.7% | 53.9% | 77.1% | 64.1% | 0.449 ± 0.163 |
| gru_motion_hr | 1 | 0.453 [0.400, 0.498] | – | 61.3% [57.5%, 64.6%] | 0.607 | 60.7% | 54.0% | 69.8% | 68.4% | 0.452 ± 0.171 |

## Paired comparison against `tcn_motion_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.026 | [-0.047, -0.004] | 0.993 |
| gru_motion_hr | -0.033 | [-0.056, -0.011] | 0.999 |
| gru_hr | -0.038 | [-0.060, -0.015] | 1.000 |
| gbdt_motion_hr | -0.140 | [-0.175, -0.104] | 1.000 |
| gbdt_hr | -0.152 | [-0.184, -0.120] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -4.1 [-51.2, +42.9] | +7.5 [-28.9, +43.8] | -1.2% [-12.1%, +9.7%] | -2.8 [-34.2, +28.7] |
| gbdt_motion_hr | -6.6 [-56.8, +43.7] | +8.6 [-37.9, +55.0] | -1.7% [-13.2%, +9.9%] | -1.9 [-33.7, +29.8] |
| tcn_hr | +3.7 [-51.4, +58.7] | -2.3 [-45.6, +41.1] | +0.7% [-11.8%, +13.2%] | -1.9 [-33.3, +29.5] |
| tcn_motion_hr | -3.3 [-60.5, +53.9] | +5.3 [-45.3, +56.0] | -0.7% [-13.5%, +12.1%] | -2.3 [-30.0, +25.5] |
| gru_hr | +0.2 [-52.7, +53.1] | +0.6 [-39.0, +40.2] | -0.0% [-12.2%, +12.1%] | -1.4 [-38.0, +35.1] |
| gru_motion_hr | -10.3 [-84.9, +64.3] | +8.3 [-51.4, +68.0] | -2.3% [-19.1%, +14.6%] | +0.8 [-37.5, +39.0] |
