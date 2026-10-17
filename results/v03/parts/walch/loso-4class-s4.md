# OpenVitals benchmark: loso / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 31 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.324 [0.282, 0.365] | – | 54.0% [51.2%, 56.7%] | 0.504 | 49.4% | 50.7% | 56.9% | 62.3% | 0.318 ± 0.114 |
| gbdt_motion_hr | 1 | 0.375 [0.335, 0.415] | – | 59.9% [57.2%, 62.5%] | 0.555 | 60.4% | 63.0% | 50.2% | 58.0% | 0.364 ± 0.121 |
| tcn_hr | 1 | 0.475 [0.425, 0.522] | – | 64.3% [60.9%, 67.5%] | 0.612 | 56.5% | 57.3% | 78.6% | 76.4% | 0.468 ± 0.137 |
| tcn_motion_hr | 1 | 0.499 [0.456, 0.539] | – | 65.9% [62.9%, 68.6%] | 0.638 | 74.2% | 58.6% | 77.8% | 73.8% | 0.485 ± 0.127 |

## Paired comparison against `tcn_motion_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.025 | [-0.058, +0.010] | 0.925 |
| gbdt_motion_hr | -0.124 | [-0.165, -0.084] | 1.000 |
| gbdt_hr | -0.176 | [-0.214, -0.136] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -12.0 [-60.2, +36.1] | +10.9 [-30.1, +52.0] | -3.0% [-15.1%, +9.1%] | +3.8 [-18.9, +26.4] |
| gbdt_motion_hr | -11.9 [-58.2, +34.3] | +13.4 [-26.9, +53.8] | -2.9% [-14.2%, +8.5%] | +1.2 [-19.7, +22.2] |
| tcn_hr | -11.6 [-72.1, +48.9] | +9.0 [-41.7, +59.8] | -2.8% [-16.1%, +10.5%] | +3.0 [-19.4, +25.5] |
| tcn_motion_hr | -24.3 [-82.3, +33.7] | +19.7 [-28.7, +68.1] | -6.2% [-21.3%, +9.0%] | +2.2 [-19.6, +23.9] |
