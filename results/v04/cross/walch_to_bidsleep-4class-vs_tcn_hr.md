# OpenVitals benchmark: walch_to_bidsleep / 4class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 47 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 5; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 5 | 0.312 [0.276, 0.345] | 0.003 | 51.8% [49.4%, 54.0%] | 0.507 | 56.3% | 51.5% | 56.0% | 47.5% | 0.304 ± 0.123 |
| gbdt_motion_hr | 5 | 0.299 [0.266, 0.329] | 0.002 | 52.3% [49.9%, 54.4%] | 0.497 | 65.0% | 61.8% | 41.7% | 37.9% | 0.292 ± 0.113 |
| tcn_hr | 5 | 0.431 [0.385, 0.472] | 0.017 | 60.8% [57.6%, 63.6%] | 0.584 | 50.6% | 62.2% | 63.0% | 60.7% | 0.424 ± 0.157 |
| tcn_motion_hr | 5 | 0.411 [0.367, 0.450] | 0.026 | 59.4% [56.4%, 62.1%] | 0.574 | 64.3% | 63.9% | 57.6% | 51.1% | 0.404 ± 0.150 |
| gru_hr | 5 | 0.417 [0.373, 0.459] | 0.015 | 59.4% [56.3%, 62.2%] | 0.578 | 60.9% | 59.5% | 61.1% | 57.5% | 0.412 ± 0.151 |
| gru_motion_hr | 5 | 0.381 [0.333, 0.424] | 0.030 | 56.7% [53.2%, 59.8%] | 0.551 | 70.4% | 59.6% | 54.9% | 47.5% | 0.376 ± 0.156 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | -0.013 | [-0.031, +0.003] | 0.942 |
| tcn_motion_hr | -0.020 | [-0.041, -0.000] | 0.978 |
| gru_motion_hr | -0.050 | [-0.078, -0.023] | 1.000 |
| gbdt_hr | -0.119 | [-0.143, -0.095] | 1.000 |
| gbdt_motion_hr | -0.132 | [-0.160, -0.103] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -16.6 [-64.9, +31.8] | +17.2 [-21.6, +56.0] | -3.9% [-15.3%, +7.4%] | +0.2 [-32.9, +33.3] |
| gbdt_motion_hr | -30.0 [-82.1, +22.0] | +27.7 [-18.4, +73.7] | -6.8% [-17.5%, +4.0%] | +0.9 [-29.1, +30.9] |
| tcn_hr | -6.9 [-55.9, +42.1] | +14.8 [-29.4, +59.0] | -1.7% [-12.7%, +9.4%] | -8.5 [-48.4, +31.4] |
| tcn_motion_hr | -26.5 [-98.5, +45.4] | +27.4 [-39.6, +94.5] | -6.0% [-21.4%, +9.4%] | -3.0 [-37.2, +31.3] |
| gru_hr | -18.2 [-65.5, +29.1] | +20.0 [-26.6, +66.6] | -4.2% [-14.7%, +6.2%] | -2.8 [-47.6, +42.0] |
| gru_motion_hr | -44.0 [-133.2, +45.1] | +35.8 [-43.8, +115.3] | -9.9% [-28.7%, +8.9%] | +2.9 [-32.5, +38.3] |
