# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 47 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.290 [0.259, 0.318] | – | 48.8% [46.7%, 50.9%] | 0.478 | 45.9% | 36.0% | 65.8% | 72.0% | 0.282 ± 0.079 |
| gbdt_motion_hr | 1 | 0.327 [0.295, 0.359] | – | 53.8% [51.4%, 56.1%] | 0.521 | 44.5% | 46.4% | 64.3% | 69.8% | 0.315 ± 0.092 |
| tcn_hr | 1 | 0.418 [0.372, 0.463] | – | 59.7% [56.5%, 62.7%] | 0.559 | 39.4% | 50.5% | 79.4% | 78.5% | 0.410 ± 0.129 |
| tcn_motion_hr | 1 | 0.441 [0.403, 0.477] | – | 60.9% [58.3%, 63.4%] | 0.579 | 50.2% | 50.5% | 80.1% | 79.5% | 0.428 ± 0.110 |
| gru_hr | 1 | 0.433 [0.396, 0.470] | – | 60.7% [58.2%, 63.2%] | 0.579 | 55.0% | 52.1% | 78.1% | 73.9% | 0.414 ± 0.127 |
| gru_motion_hr | 1 | 0.425 [0.379, 0.468] | – | 59.8% [56.7%, 62.6%] | 0.581 | 52.5% | 49.1% | 81.3% | 76.5% | 0.405 ± 0.134 |

## Paired comparison against `tcn_motion_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | -0.008 | [-0.033, +0.018] | 0.714 |
| gru_motion_hr | -0.016 | [-0.039, +0.008] | 0.904 |
| tcn_hr | -0.022 | [-0.042, -0.002] | 0.986 |
| gbdt_motion_hr | -0.113 | [-0.142, -0.085] | 1.000 |
| gbdt_hr | -0.151 | [-0.173, -0.127] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.0 [-48.4, +42.3] | +5.2 [-33.2, +43.6] | -1.2% [-13.8%, +11.4%] | +0.6 [-17.7, +19.0] |
| gbdt_motion_hr | +6.7 [-28.5, +41.9] | -2.9 [-33.9, +28.0] | +1.5% [-8.7%, +11.6%] | -1.1 [-22.0, +19.8] |
| tcn_hr | +4.7 [-40.3, +49.7] | -0.2 [-33.0, +32.6] | +0.8% [-10.7%, +12.4%] | -2.4 [-24.3, +19.4] |
| tcn_motion_hr | -7.9 [-54.2, +38.4] | +10.3 [-20.8, +41.5] | -2.1% [-13.2%, +9.1%] | -0.2 [-23.5, +23.0] |
| gru_hr | -14.1 [-64.3, +36.0] | +9.1 [-30.8, +49.1] | -4.2% [-19.6%, +11.1%] | +6.6 [-24.6, +37.8] |
| gru_motion_hr | -0.7 [-41.5, +40.1] | -3.8 [-33.7, +26.1] | -0.8% [-12.6%, +11.0%] | +6.8 [-19.8, +33.4] |
