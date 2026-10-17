# OpenVitals benchmark: loso / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 31 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.325 [0.285, 0.366] | – | 54.1% [51.4%, 56.7%] | 0.505 | 49.9% | 50.8% | 57.5% | 61.9% | 0.318 ± 0.114 |
| gbdt_motion_hr | 1 | 0.373 [0.333, 0.410] | – | 59.8% [57.2%, 62.2%] | 0.552 | 59.6% | 62.9% | 49.6% | 58.2% | 0.363 ± 0.115 |
| tcn_hr | 1 | 0.478 [0.430, 0.521] | – | 65.0% [61.8%, 67.8%] | 0.616 | 55.2% | 60.0% | 78.2% | 73.2% | 0.471 ± 0.132 |
| tcn_motion_hr | 1 | 0.505 [0.466, 0.544] | – | 67.2% [64.7%, 69.6%] | 0.647 | 68.0% | 62.3% | 72.4% | 76.0% | 0.496 ± 0.120 |

## Paired comparison against `tcn_motion_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.027 | [-0.060, +0.004] | 0.952 |
| gbdt_motion_hr | -0.133 | [-0.172, -0.098] | 1.000 |
| gbdt_hr | -0.180 | [-0.215, -0.146] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -12.5 [-61.3, +36.2] | +11.7 [-28.8, +52.3] | -3.2% [-15.4%, +9.0%] | +3.5 [-18.3, +25.3] |
| gbdt_motion_hr | -11.2 [-56.8, +34.5] | +12.8 [-25.4, +50.9] | -2.7% [-14.0%, +8.5%] | +0.8 [-20.4, +22.0] |
| tcn_hr | -7.8 [-64.1, +48.4] | +6.0 [-45.2, +57.2] | -2.0% [-14.4%, +10.4%] | +2.4 [-14.1, +18.8] |
| tcn_motion_hr | -12.1 [-55.3, +31.0] | +10.1 [-26.3, +46.6] | -3.2% [-14.3%, +7.9%] | +2.0 [-15.3, +19.3] |
