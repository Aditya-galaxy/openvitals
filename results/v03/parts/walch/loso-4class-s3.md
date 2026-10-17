# OpenVitals benchmark: loso / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 31 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.328 [0.287, 0.367] | – | 54.3% [51.7%, 56.9%] | 0.508 | 50.6% | 51.1% | 56.8% | 62.4% | 0.323 ± 0.113 |
| gbdt_motion_hr | 1 | 0.373 [0.332, 0.413] | – | 59.8% [57.1%, 62.4%] | 0.553 | 59.8% | 62.8% | 49.8% | 58.3% | 0.363 ± 0.118 |
| tcn_hr | 1 | 0.479 [0.430, 0.524] | – | 65.5% [62.4%, 68.3%] | 0.623 | 61.1% | 62.0% | 75.5% | 69.7% | 0.462 ± 0.153 |
| tcn_motion_hr | 1 | 0.500 [0.457, 0.540] | – | 67.2% [64.4%, 69.8%] | 0.640 | 68.1% | 64.7% | 67.0% | 73.6% | 0.482 ± 0.127 |

## Paired comparison against `tcn_motion_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.021 | [-0.051, +0.012] | 0.885 |
| gbdt_motion_hr | -0.127 | [-0.164, -0.092] | 1.000 |
| gbdt_hr | -0.173 | [-0.208, -0.135] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -11.6 [-59.1, +35.9] | +11.5 [-27.8, +50.7] | -2.8% [-14.7%, +9.1%] | +2.9 [-17.9, +23.7] |
| gbdt_motion_hr | -11.6 [-58.3, +35.0] | +13.1 [-28.2, +54.4] | -2.9% [-14.2%, +8.4%] | +1.3 [-19.5, +22.0] |
| tcn_hr | -11.5 [-51.8, +28.8] | +9.5 [-27.1, +46.1] | -3.4% [-15.1%, +8.3%] | +3.4 [-17.8, +24.6] |
| tcn_motion_hr | -16.9 [-64.5, +30.7] | +15.2 [-29.6, +60.1] | -4.6% [-17.5%, +8.4%] | +2.3 [-20.5, +25.1] |
