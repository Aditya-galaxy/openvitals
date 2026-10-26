# OpenVitals benchmark: walch_to_bidsleep / 4class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 47 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.308 [0.273, 0.341] | – | 51.6% [49.2%, 53.8%] | 0.504 | 56.1% | 51.8% | 55.8% | 46.5% | 0.300 ± 0.120 |
| gbdt_motion_hr | 1 | 0.297 [0.262, 0.328] | – | 52.2% [49.8%, 54.4%] | 0.495 | 65.2% | 62.3% | 40.3% | 37.9% | 0.289 ± 0.119 |
| tcn_hr | 1 | 0.422 [0.379, 0.462] | – | 59.9% [56.8%, 62.6%] | 0.577 | 63.5% | 62.8% | 58.4% | 54.3% | 0.415 ± 0.158 |
| tcn_motion_hr | 1 | 0.396 [0.347, 0.442] | – | 59.2% [55.9%, 62.4%] | 0.562 | 63.1% | 70.0% | 53.0% | 43.2% | 0.388 ± 0.170 |
| gru_hr | 1 | 0.405 [0.362, 0.445] | – | 57.9% [54.8%, 60.6%] | 0.563 | 60.1% | 55.4% | 71.6% | 51.4% | 0.399 ± 0.146 |
| gru_motion_hr | 1 | 0.349 [0.296, 0.397] | – | 56.7% [53.2%, 59.9%] | 0.531 | 66.8% | 71.8% | 40.2% | 38.2% | 0.339 ± 0.174 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | -0.017 | [-0.038, +0.004] | 0.946 |
| tcn_motion_hr | -0.026 | [-0.056, -0.000] | 0.978 |
| gru_motion_hr | -0.073 | [-0.107, -0.042] | 1.000 |
| gbdt_hr | -0.114 | [-0.139, -0.087] | 1.000 |
| gbdt_motion_hr | -0.125 | [-0.158, -0.095] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -17.1 [-67.0, +32.7] | +17.9 [-24.3, +60.0] | -4.1% [-15.8%, +7.7%] | +0.0 [-32.9, +33.0] |
| gbdt_motion_hr | -29.7 [-81.0, +21.6] | +26.3 [-20.0, +72.6] | -6.7% [-17.3%, +3.9%] | +1.0 [-29.0, +31.0] |
| tcn_hr | -30.9 [-81.5, +19.7] | +32.1 [-15.5, +79.7] | -7.3% [-18.6%, +4.0%] | -2.9 [-39.8, +33.9] |
| tcn_motion_hr | -25.9 [-97.8, +46.0] | +29.0 [-37.7, +95.6] | -5.8% [-20.9%, +9.3%] | -6.1 [-40.4, +28.2] |
| gru_hr | -23.1 [-74.5, +28.2] | +30.9 [-18.2, +80.0] | -5.4% [-16.9%, +6.0%] | -9.1 [-56.2, +38.0] |
| gru_motion_hr | -30.6 [-104.8, +43.7] | +27.1 [-39.7, +93.8] | -6.8% [-22.5%, +8.9%] | -0.9 [-39.4, +37.6] |
