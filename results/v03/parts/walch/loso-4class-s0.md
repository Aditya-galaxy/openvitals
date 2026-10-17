# OpenVitals benchmark: loso / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 31 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.326 [0.287, 0.365] | – | 54.1% [51.6%, 56.7%] | 0.506 | 50.0% | 50.8% | 57.1% | 62.4% | 0.321 ± 0.108 |
| gbdt_motion_hr | 1 | 0.376 [0.337, 0.415] | – | 60.0% [57.4%, 62.5%] | 0.555 | 59.6% | 63.0% | 50.2% | 58.5% | 0.364 ± 0.117 |
| tcn_hr | 1 | 0.511 [0.464, 0.553] | – | 67.6% [64.5%, 70.3%] | 0.646 | 60.7% | 62.8% | 76.3% | 77.0% | 0.501 ± 0.124 |
| tcn_motion_hr | 1 | 0.500 [0.456, 0.544] | – | 66.9% [64.1%, 69.9%] | 0.646 | 66.8% | 62.9% | 74.9% | 72.2% | 0.482 ± 0.142 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | -0.011 | [-0.045, +0.022] | 0.757 |
| gbdt_motion_hr | -0.135 | [-0.177, -0.092] | 1.000 |
| gbdt_hr | -0.185 | [-0.223, -0.148] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -11.8 [-61.0, +37.3] | +11.7 [-29.2, +52.5] | -2.9% [-15.3%, +9.4%] | +3.0 [-17.2, +23.1] |
| gbdt_motion_hr | -11.9 [-59.7, +35.8] | +13.5 [-28.1, +55.0] | -3.0% [-14.9%, +9.0%] | +1.2 [-19.6, +22.0] |
| tcn_hr | -9.2 [-53.0, +34.7] | +9.7 [-30.8, +50.2] | -1.9% [-12.2%, +8.3%] | +2.2 [-16.2, +20.7] |
| tcn_motion_hr | -9.5 [-47.9, +28.9] | +7.4 [-27.1, +41.8] | -2.9% [-14.3%, +8.5%] | +2.9 [-19.4, +25.3] |
