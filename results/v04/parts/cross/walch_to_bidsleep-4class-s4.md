# OpenVitals benchmark: walch_to_bidsleep / 4class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 47 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.312 [0.276, 0.345] | – | 51.8% [49.4%, 54.0%] | 0.507 | 56.2% | 51.4% | 55.8% | 48.0% | 0.304 ± 0.122 |
| gbdt_motion_hr | 1 | 0.299 [0.265, 0.329] | – | 52.1% [49.8%, 54.2%] | 0.496 | 65.0% | 61.4% | 42.7% | 37.6% | 0.292 ± 0.113 |
| tcn_hr | 1 | 0.442 [0.395, 0.484] | – | 61.3% [58.0%, 64.1%] | 0.591 | 55.6% | 61.3% | 61.5% | 63.5% | 0.437 ± 0.162 |
| tcn_motion_hr | 1 | 0.435 [0.389, 0.480] | – | 60.1% [56.7%, 63.3%] | 0.588 | 69.7% | 57.9% | 65.8% | 56.0% | 0.430 ± 0.159 |
| gru_hr | 1 | 0.419 [0.374, 0.462] | – | 59.1% [55.9%, 62.0%] | 0.575 | 62.2% | 57.6% | 63.7% | 57.3% | 0.416 ± 0.157 |
| gru_motion_hr | 1 | 0.401 [0.353, 0.444] | – | 57.6% [54.3%, 60.7%] | 0.566 | 68.9% | 55.3% | 61.1% | 54.8% | 0.398 ± 0.160 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | -0.007 | [-0.032, +0.017] | 0.728 |
| gru_hr | -0.023 | [-0.047, +0.000] | 0.974 |
| gru_motion_hr | -0.041 | [-0.072, -0.012] | 0.998 |
| gbdt_hr | -0.130 | [-0.159, -0.102] | 1.000 |
| gbdt_motion_hr | -0.144 | [-0.177, -0.110] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -16.5 [-65.1, +32.1] | +17.3 [-19.0, +53.6] | -3.9% [-15.4%, +7.6%] | -0.0 [-33.4, +33.4] |
| gbdt_motion_hr | -30.4 [-82.8, +22.1] | +27.5 [-20.9, +75.8] | -6.9% [-17.6%, +3.9%] | +0.6 [-29.0, +30.1] |
| tcn_hr | -12.7 [-58.2, +32.8] | +17.2 [-26.0, +60.5] | -3.0% [-13.2%, +7.1%] | -5.7 [-47.0, +35.6] |
| tcn_motion_hr | -35.3 [-115.7, +45.0] | +31.1 [-47.1, +109.4] | -8.0% [-24.7%, +8.7%] | +1.8 [-29.5, +33.2] |
| gru_hr | -25.1 [-72.9, +22.7] | +26.6 [-19.8, +72.9] | -5.7% [-16.2%, +4.7%] | -3.2 [-47.0, +40.5] |
| gru_motion_hr | -33.4 [-115.5, +48.7] | +29.2 [-46.7, +105.1] | -7.5% [-24.7%, +9.7%] | +2.5 [-30.2, +35.3] |
