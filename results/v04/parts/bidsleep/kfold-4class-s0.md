# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 47 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.333 [0.299, 0.364] | – | 52.2% [49.6%, 54.5%] | 0.526 | 54.9% | 41.7% | 63.9% | 61.2% | 0.330 ± 0.115 |
| gbdt_motion_hr | 1 | 0.350 [0.315, 0.382] | – | 54.4% [51.9%, 56.8%] | 0.547 | 58.7% | 49.8% | 60.4% | 56.6% | 0.345 ± 0.119 |
| tcn_hr | 1 | 0.472 [0.421, 0.516] | – | 62.8% [59.2%, 65.8%] | 0.624 | 58.7% | 55.4% | 75.7% | 68.2% | 0.470 ± 0.169 |
| tcn_motion_hr | 1 | 0.460 [0.412, 0.502] | – | 62.3% [58.9%, 65.4%] | 0.616 | 59.4% | 58.5% | 75.1% | 60.8% | 0.458 ± 0.168 |
| gru_hr | 1 | 0.450 [0.400, 0.497] | – | 60.8% [57.0%, 64.3%] | 0.604 | 62.4% | 52.1% | 74.4% | 65.7% | 0.453 ± 0.157 |
| gru_motion_hr | 1 | 0.439 [0.390, 0.485] | – | 60.0% [56.4%, 63.3%] | 0.597 | 66.5% | 51.5% | 73.0% | 63.0% | 0.442 ± 0.166 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | -0.012 | [-0.033, +0.007] | 0.883 |
| gru_hr | -0.022 | [-0.047, +0.002] | 0.958 |
| gru_motion_hr | -0.033 | [-0.057, -0.012] | 0.998 |
| gbdt_motion_hr | -0.122 | [-0.154, -0.092] | 1.000 |
| gbdt_hr | -0.139 | [-0.162, -0.116] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -4.4 [-52.2, +43.4] | +8.1 [-28.6, +44.9] | -1.3% [-12.4%, +9.8%] | -3.2 [-33.4, +27.0] |
| gbdt_motion_hr | -5.6 [-55.1, +43.9] | +8.8 [-33.1, +50.6] | -1.4% [-12.9%, +10.0%] | -3.0 [-32.7, +26.6] |
| tcn_hr | -2.7 [-53.3, +47.9] | +2.2 [-40.5, +44.8] | -0.8% [-12.5%, +10.9%] | +0.1 [-32.8, +33.1] |
| tcn_motion_hr | -4.9 [-66.3, +56.4] | +5.9 [-50.1, +61.9] | -1.4% [-15.1%, +12.4%] | -1.5 [-30.9, +27.9] |
| gru_hr | -11.3 [-66.3, +43.7] | +4.8 [-44.2, +53.8] | -2.9% [-15.2%, +9.5%] | +5.5 [-25.9, +36.9] |
| gru_motion_hr | -17.1 [-94.1, +59.9] | +12.2 [-57.7, +82.2] | -3.8% [-19.7%, +12.1%] | +3.9 [-30.4, +38.2] |
