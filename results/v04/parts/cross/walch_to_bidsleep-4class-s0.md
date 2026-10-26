# OpenVitals benchmark: walch_to_bidsleep / 4class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 47 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.310 [0.274, 0.344] | – | 51.7% [49.3%, 54.0%] | 0.506 | 55.8% | 51.6% | 56.4% | 47.0% | 0.303 ± 0.124 |
| gbdt_motion_hr | 1 | 0.298 [0.266, 0.326] | – | 52.2% [49.9%, 54.2%] | 0.495 | 65.1% | 61.6% | 40.8% | 38.6% | 0.291 ± 0.110 |
| tcn_hr | 1 | 0.435 [0.390, 0.476] | – | 59.8% [56.7%, 62.8%] | 0.585 | 55.3% | 53.3% | 73.2% | 63.6% | 0.429 ± 0.156 |
| tcn_motion_hr | 1 | 0.375 [0.332, 0.414] | – | 56.5% [53.5%, 59.3%] | 0.548 | 71.7% | 60.7% | 50.7% | 47.2% | 0.369 ± 0.149 |
| gru_hr | 1 | 0.411 [0.366, 0.454] | – | 59.2% [56.0%, 62.1%] | 0.577 | 64.7% | 61.1% | 59.7% | 53.2% | 0.403 ± 0.157 |
| gru_motion_hr | 1 | 0.422 [0.371, 0.464] | – | 59.3% [55.6%, 62.2%] | 0.582 | 66.9% | 56.3% | 62.3% | 59.3% | 0.418 ± 0.160 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_motion_hr | -0.012 | [-0.042, +0.017] | 0.789 |
| gru_hr | -0.024 | [-0.046, -0.002] | 0.984 |
| tcn_motion_hr | -0.060 | [-0.089, -0.030] | 1.000 |
| gbdt_hr | -0.124 | [-0.149, -0.099] | 1.000 |
| gbdt_motion_hr | -0.137 | [-0.168, -0.106] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -15.9 [-63.5, +31.7] | +16.1 [-23.5, +55.8] | -3.8% [-14.8%, +7.3%] | +0.6 [-31.5, +32.8] |
| gbdt_motion_hr | -30.5 [-81.7, +20.7] | +28.6 [-14.4, +71.5] | -6.9% [-17.4%, +3.7%] | +0.8 [-28.2, +29.8] |
| tcn_hr | -12.0 [-61.9, +38.0] | +17.5 [-28.7, +63.7] | -2.9% [-14.2%, +8.4%] | -5.5 [-48.0, +37.0] |
| tcn_motion_hr | -47.3 [-128.0, +33.4] | +43.9 [-30.4, +118.1] | -11.1% [-29.2%, +7.0%] | +2.4 [-29.7, +34.5] |
| gru_hr | -21.2 [-64.8, +22.5] | +19.5 [-29.5, +68.5] | -5.0% [-14.6%, +4.5%] | +1.2 [-44.3, +46.7] |
| gru_motion_hr | -29.1 [-101.9, +43.7] | +22.5 [-36.9, +82.0] | -6.5% [-22.4%, +9.4%] | +3.1 [-35.6, +41.9] |
