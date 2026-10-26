# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 47 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.336 [0.304, 0.366] | – | 52.5% [50.1%, 54.8%] | 0.528 | 54.2% | 42.5% | 63.2% | 61.8% | 0.333 ± 0.111 |
| gbdt_motion_hr | 1 | 0.346 [0.310, 0.379] | – | 54.1% [51.6%, 56.5%] | 0.544 | 57.4% | 49.7% | 60.1% | 56.4% | 0.342 ± 0.123 |
| tcn_hr | 1 | 0.466 [0.414, 0.510] | – | 62.4% [58.7%, 65.7%] | 0.616 | 54.6% | 55.1% | 73.8% | 70.1% | 0.465 ± 0.169 |
| tcn_motion_hr | 1 | 0.460 [0.406, 0.510] | – | 62.4% [58.7%, 65.9%] | 0.618 | 57.2% | 58.4% | 71.4% | 65.0% | 0.458 ± 0.184 |
| gru_hr | 1 | 0.434 [0.379, 0.481] | – | 60.2% [56.2%, 63.5%] | 0.598 | 59.0% | 54.6% | 72.5% | 61.6% | 0.435 ± 0.170 |
| gru_motion_hr | 1 | 0.442 [0.389, 0.490] | – | 60.4% [56.6%, 63.8%] | 0.605 | 66.3% | 52.7% | 74.5% | 61.4% | 0.443 ± 0.174 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | -0.006 | [-0.029, +0.015] | 0.707 |
| gru_motion_hr | -0.024 | [-0.051, +0.001] | 0.970 |
| gru_hr | -0.032 | [-0.052, -0.013] | 0.998 |
| gbdt_motion_hr | -0.120 | [-0.155, -0.083] | 1.000 |
| gbdt_hr | -0.129 | [-0.156, -0.102] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.7 [-51.0, +43.7] | +7.0 [-30.6, +44.6] | -1.1% [-12.1%, +9.9%] | -2.9 [-33.6, +27.7] |
| gbdt_motion_hr | -4.8 [-53.4, +43.7] | +7.8 [-35.4, +51.0] | -1.2% [-12.2%, +9.7%] | -2.6 [-32.8, +27.6] |
| tcn_hr | +0.7 [-54.2, +55.7] | +0.4 [-47.7, +48.5] | +0.1% [-11.9%, +12.0%] | -1.3 [-34.7, +32.1] |
| tcn_motion_hr | +0.8 [-60.4, +61.9] | -0.5 [-52.1, +51.0] | +0.2% [-13.0%, +13.5%] | -0.4 [-34.0, +33.3] |
| gru_hr | -4.7 [-55.9, +46.5] | +0.3 [-43.5, +44.1] | -1.2% [-12.9%, +10.5%] | +3.4 [-24.7, +31.4] |
| gru_motion_hr | -12.4 [-76.8, +52.1] | +8.6 [-49.2, +66.3] | -2.7% [-16.4%, +11.0%] | +3.5 [-25.5, +32.4] |
