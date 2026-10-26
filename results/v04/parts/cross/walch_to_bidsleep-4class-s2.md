# OpenVitals benchmark: walch_to_bidsleep / 4class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 47 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.315 [0.279, 0.347] | – | 52.1% [49.6%, 54.3%] | 0.510 | 56.6% | 51.9% | 55.8% | 47.9% | 0.307 ± 0.123 |
| gbdt_motion_hr | 1 | 0.302 [0.267, 0.333] | – | 52.4% [50.0%, 54.6%] | 0.499 | 64.8% | 61.9% | 43.2% | 37.6% | 0.294 ± 0.118 |
| tcn_hr | 1 | 0.447 [0.401, 0.490] | – | 61.8% [58.6%, 64.9%] | 0.596 | 50.0% | 62.3% | 70.8% | 59.3% | 0.439 ± 0.162 |
| tcn_motion_hr | 1 | 0.434 [0.390, 0.474] | – | 60.7% [57.6%, 63.6%] | 0.593 | 58.5% | 60.4% | 69.1% | 56.1% | 0.428 ± 0.155 |
| gru_hr | 1 | 0.408 [0.359, 0.450] | – | 60.1% [56.8%, 62.9%] | 0.571 | 53.9% | 67.0% | 47.5% | 59.3% | 0.399 ± 0.162 |
| gru_motion_hr | 1 | 0.362 [0.309, 0.411] | – | 55.6% [51.9%, 59.2%] | 0.536 | 70.6% | 61.6% | 53.6% | 40.7% | 0.357 ± 0.173 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | -0.013 | [-0.035, +0.009] | 0.871 |
| gru_hr | -0.039 | [-0.069, -0.011] | 0.996 |
| gru_motion_hr | -0.085 | [-0.125, -0.049] | 1.000 |
| gbdt_hr | -0.132 | [-0.162, -0.102] | 1.000 |
| gbdt_motion_hr | -0.146 | [-0.178, -0.113] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -16.1 [-64.3, +32.1] | +16.8 [-21.2, +54.7] | -3.9% [-15.2%, +7.5%] | +0.2 [-33.6, +34.0] |
| gbdt_motion_hr | -29.5 [-81.7, +22.7] | +28.2 [-18.5, +75.0] | -6.7% [-17.5%, +4.1%] | +1.1 [-29.9, +32.1] |
| tcn_hr | -3.4 [-50.9, +44.1] | +12.1 [-31.5, +55.6] | -0.9% [-11.6%, +9.9%] | -9.7 [-50.1, +30.7] |
| tcn_motion_hr | -9.1 [-66.5, +48.3] | +14.7 [-38.7, +68.0] | -2.0% [-14.4%, +10.3%] | -6.4 [-43.1, +30.4] |
| gru_hr | -8.8 [-61.1, +43.6] | +14.1 [-26.8, +55.0] | -2.2% [-14.1%, +9.8%] | -6.6 [-47.2, +34.1] |
| gru_motion_hr | -50.2 [-143.4, +43.0] | +41.5 [-43.9, +126.8] | -11.4% [-31.5%, +8.7%] | +0.3 [-35.6, +36.2] |
