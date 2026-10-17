# OpenVitals benchmark: kfold / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.310 [0.270, 0.349] | – | 53.4% [51.0%, 55.9%] | 0.496 | 49.2% | 51.5% | 52.5% | 60.6% | 0.306 ± 0.112 |
| gbdt_motion_hr | 1 | 0.374 [0.331, 0.416] | – | 60.5% [57.8%, 63.0%] | 0.554 | 59.6% | 65.6% | 45.0% | 57.6% | 0.361 ± 0.127 |
| tcn_hr | 1 | 0.490 [0.441, 0.536] | – | 66.6% [63.4%, 69.5%] | 0.627 | 55.8% | 64.4% | 73.5% | 71.9% | 0.476 ± 0.135 |
| tcn_motion_hr | 1 | 0.472 [0.431, 0.507] | – | 64.6% [61.9%, 66.9%] | 0.624 | 66.7% | 59.5% | 74.6% | 70.5% | 0.461 ± 0.113 |
| gru_hr | 1 | 0.410 [0.365, 0.452] | – | 59.4% [56.5%, 62.1%] | 0.566 | 59.4% | 53.0% | 72.9% | 67.5% | 0.395 ± 0.135 |
| gru_motion_hr | 1 | 0.457 [0.405, 0.503] | – | 63.8% [60.6%, 66.7%] | 0.610 | 67.3% | 60.1% | 67.2% | 69.6% | 0.440 ± 0.145 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | -0.018 | [-0.045, +0.009] | 0.912 |
| gru_motion_hr | -0.033 | [-0.065, -0.002] | 0.983 |
| gru_hr | -0.080 | [-0.110, -0.050] | 1.000 |
| gbdt_motion_hr | -0.116 | [-0.162, -0.071] | 1.000 |
| gbdt_hr | -0.180 | [-0.218, -0.141] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -10.9 [-61.8, +40.0] | +11.4 [-30.5, +53.3] | -2.7% [-15.0%, +9.6%] | +2.3 [-17.8, +22.5] |
| gbdt_motion_hr | -9.4 [-56.7, +37.9] | +10.8 [-29.6, +51.2] | -2.3% [-14.1%, +9.5%] | +1.2 [-20.4, +22.9] |
| tcn_hr | -5.8 [-60.0, +48.4] | +6.5 [-40.5, +53.6] | -1.4% [-13.8%, +11.0%] | +1.3 [-19.0, +21.6] |
| tcn_motion_hr | -11.5 [-57.8, +34.9] | +12.9 [-31.3, +57.0] | -3.1% [-14.7%, +8.5%] | +0.5 [-21.1, +22.1] |
| gru_hr | -22.1 [-96.2, +51.9] | +18.5 [-42.0, +78.9] | -5.7% [-23.3%, +12.0%] | +4.3 [-18.1, +26.7] |
| gru_motion_hr | -14.9 [-69.3, +39.5] | +14.6 [-30.6, +59.9] | -3.4% [-16.6%, +9.7%] | +2.0 [-21.9, +25.8] |
