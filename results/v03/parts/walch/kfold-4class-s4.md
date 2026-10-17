# OpenVitals benchmark: kfold / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.310 [0.271, 0.348] | – | 53.5% [51.1%, 55.8%] | 0.495 | 48.8% | 51.8% | 52.5% | 60.2% | 0.305 ± 0.109 |
| gbdt_motion_hr | 1 | 0.363 [0.322, 0.402] | – | 59.7% [57.1%, 62.2%] | 0.547 | 59.4% | 64.7% | 45.0% | 56.5% | 0.351 ± 0.118 |
| tcn_hr | 1 | 0.462 [0.413, 0.507] | – | 63.9% [60.8%, 66.7%] | 0.607 | 59.8% | 59.7% | 75.0% | 69.2% | 0.459 ± 0.134 |
| tcn_motion_hr | 1 | 0.454 [0.399, 0.507] | – | 64.4% [60.6%, 67.8%] | 0.620 | 64.9% | 62.4% | 65.9% | 68.3% | 0.443 ± 0.150 |
| gru_hr | 1 | 0.442 [0.383, 0.495] | – | 62.4% [58.5%, 66.0%] | 0.588 | 57.7% | 58.3% | 70.7% | 69.5% | 0.431 ± 0.148 |
| gru_motion_hr | 1 | 0.428 [0.369, 0.481] | – | 61.3% [57.3%, 64.9%] | 0.590 | 68.0% | 55.8% | 65.3% | 70.2% | 0.415 ± 0.148 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | -0.008 | [-0.041, +0.021] | 0.697 |
| gru_hr | -0.021 | [-0.060, +0.018] | 0.854 |
| gru_motion_hr | -0.034 | [-0.074, +0.003] | 0.965 |
| gbdt_motion_hr | -0.099 | [-0.144, -0.057] | 1.000 |
| gbdt_hr | -0.153 | [-0.190, -0.115] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -11.4 [-64.5, +41.7] | +12.3 [-30.2, +54.9] | -2.7% [-15.2%, +9.8%] | +1.8 [-18.5, +22.1] |
| gbdt_motion_hr | -9.3 [-54.9, +36.2] | +10.6 [-29.4, +50.7] | -2.3% [-13.5%, +8.8%] | +1.3 [-19.9, +22.5] |
| tcn_hr | -12.1 [-63.2, +38.9] | +12.8 [-33.5, +59.1] | -2.9% [-13.8%, +8.1%] | +0.7 [-20.3, +21.8] |
| tcn_motion_hr | -6.5 [-47.5, +34.4] | +7.9 [-30.2, +46.0] | -1.7% [-12.7%, +9.3%] | +0.9 [-19.5, +21.2] |
| gru_hr | -15.9 [-67.7, +36.0] | +9.8 [-38.3, +57.8] | -3.8% [-15.3%, +7.6%] | +6.9 [-25.1, +38.9] |
| gru_motion_hr | -19.3 [-80.2, +41.6] | +17.7 [-29.5, +64.8] | -4.9% [-24.1%, +14.2%] | +3.8 [-19.5, +27.1] |
