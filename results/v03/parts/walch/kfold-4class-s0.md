# OpenVitals benchmark: kfold / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.309 [0.270, 0.348] | – | 53.4% [51.0%, 55.8%] | 0.494 | 48.8% | 51.5% | 52.3% | 60.7% | 0.303 ± 0.110 |
| gbdt_motion_hr | 1 | 0.371 [0.328, 0.414] | – | 60.4% [57.6%, 63.0%] | 0.551 | 58.3% | 65.6% | 44.7% | 57.8% | 0.359 ± 0.126 |
| tcn_hr | 1 | 0.484 [0.430, 0.534] | – | 65.9% [62.2%, 69.3%] | 0.619 | 54.4% | 62.4% | 69.4% | 77.1% | 0.471 ± 0.145 |
| tcn_motion_hr | 1 | 0.486 [0.438, 0.533] | – | 66.0% [62.5%, 69.4%] | 0.633 | 64.7% | 61.8% | 66.3% | 77.0% | 0.471 ± 0.137 |
| gru_hr | 1 | 0.410 [0.359, 0.458] | – | 59.7% [56.3%, 63.1%] | 0.560 | 54.5% | 54.2% | 64.4% | 72.9% | 0.399 ± 0.140 |
| gru_motion_hr | 1 | 0.430 [0.380, 0.474] | – | 61.5% [58.2%, 64.3%] | 0.593 | 65.0% | 55.6% | 66.5% | 71.9% | 0.413 ± 0.142 |

## Paired comparison against `tcn_motion_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.001 | [-0.030, +0.029] | 0.521 |
| gru_motion_hr | -0.056 | [-0.090, -0.023] | 1.000 |
| gru_hr | -0.076 | [-0.114, -0.035] | 1.000 |
| gbdt_motion_hr | -0.114 | [-0.151, -0.076] | 1.000 |
| gbdt_hr | -0.176 | [-0.212, -0.140] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -10.9 [-63.7, +41.9] | +11.6 [-31.8, +55.0] | -2.7% [-15.2%, +9.9%] | +1.9 [-18.0, +21.8] |
| gbdt_motion_hr | -8.8 [-54.3, +36.6] | +10.3 [-28.7, +49.3] | -2.2% [-13.5%, +9.1%] | +1.2 [-20.0, +22.3] |
| tcn_hr | -7.0 [-48.7, +34.8] | +6.5 [-32.2, +45.3] | -2.0% [-12.2%, +8.2%] | +1.6 [-17.8, +21.0] |
| tcn_motion_hr | -8.5 [-49.1, +32.1] | +7.7 [-29.1, +44.5] | -2.7% [-13.6%, +8.2%] | +1.5 [-16.9, +19.8] |
| gru_hr | -21.8 [-77.4, +33.8] | +19.3 [-22.6, +61.3] | -5.2% [-20.5%, +10.1%] | +3.5 [-29.5, +36.5] |
| gru_motion_hr | -14.4 [-75.6, +46.8] | +12.2 [-36.9, +61.3] | -3.4% [-17.3%, +10.4%] | +3.7 [-23.3, +30.8] |
