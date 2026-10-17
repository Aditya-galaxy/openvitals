# OpenVitals benchmark: loso / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 31 (subject-wise; no subject in both train and test)
- Seeds per model: 5; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 5 | 0.326 [0.286, 0.365] | 0.001 | 54.1% [51.5%, 56.8%] | 0.506 | 49.9% | 50.8% | 57.1% | 62.4% | 0.320 ± 0.111 |
| gbdt_motion_hr | 5 | 0.374 [0.334, 0.413] | 0.002 | 59.8% [57.2%, 62.4%] | 0.553 | 59.8% | 63.0% | 49.7% | 58.2% | 0.363 ± 0.117 |
| tcn_hr | 5 | 0.488 [0.442, 0.530] | 0.015 | 65.7% [62.7%, 68.5%] | 0.626 | 58.5% | 60.5% | 77.0% | 74.8% | 0.478 ± 0.127 |
| tcn_motion_hr | 5 | 0.498 [0.459, 0.535] | 0.007 | 66.6% [64.2%, 68.9%] | 0.640 | 68.9% | 62.1% | 72.9% | 73.2% | 0.484 ± 0.117 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | +0.010 | [-0.013, +0.033] | 0.189 |
| gbdt_motion_hr | -0.114 | [-0.157, -0.072] | 1.000 |
| gbdt_hr | -0.162 | [-0.197, -0.129] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -11.9 [-60.3, +36.5] | +11.3 [-29.2, +51.8] | -3.0% [-15.1%, +9.1%] | +3.4 [-18.3, +25.0] |
| gbdt_motion_hr | -11.8 [-58.4, +34.8] | +13.3 [-27.2, +53.8] | -2.9% [-14.3%, +8.6%] | +1.1 [-19.9, +22.0] |
| tcn_hr | -9.6 [-60.5, +41.3] | +8.2 [-37.3, +53.7] | -2.4% [-14.4%, +9.5%] | +2.7 [-16.2, +21.6] |
| tcn_motion_hr | -15.4 [-63.2, +32.4] | +13.1 [-28.9, +55.1] | -4.0% [-16.5%, +8.4%] | +2.2 [-18.7, +23.2] |
