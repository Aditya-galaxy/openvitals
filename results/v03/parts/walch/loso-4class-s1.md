# OpenVitals benchmark: loso / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 31 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.326 [0.288, 0.366] | – | 54.1% [51.5%, 56.8%] | 0.506 | 49.6% | 50.8% | 57.2% | 62.7% | 0.321 ± 0.108 |
| gbdt_motion_hr | 1 | 0.372 [0.330, 0.413] | – | 59.7% [57.0%, 62.3%] | 0.550 | 59.5% | 63.1% | 48.9% | 58.0% | 0.360 ± 0.119 |
| tcn_hr | 1 | 0.497 [0.450, 0.542] | – | 66.3% [63.1%, 69.2%] | 0.633 | 59.3% | 60.4% | 76.5% | 77.7% | 0.490 ± 0.128 |
| tcn_motion_hr | 1 | 0.487 [0.429, 0.537] | – | 65.9% [61.9%, 69.1%] | 0.631 | 67.5% | 62.2% | 72.4% | 70.5% | 0.481 ± 0.139 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | -0.010 | [-0.052, +0.030] | 0.684 |
| gbdt_motion_hr | -0.125 | [-0.173, -0.078] | 1.000 |
| gbdt_hr | -0.170 | [-0.210, -0.131] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -11.6 [-60.2, +37.0] | +10.8 [-30.1, +51.6] | -2.9% [-15.0%, +9.2%] | +3.6 [-19.2, +26.5] |
| gbdt_motion_hr | -12.2 [-59.0, +34.5] | +13.9 [-27.3, +55.1] | -2.9% [-14.2%, +8.5%] | +0.9 [-20.2, +22.0] |
| tcn_hr | -8.0 [-61.7, +45.8] | +6.9 [-41.6, +55.4] | -1.9% [-14.0%, +10.2%] | +2.5 [-13.3, +18.4] |
| tcn_motion_hr | -14.2 [-66.0, +37.6] | +13.1 [-32.9, +59.1] | -3.4% [-15.1%, +8.2%] | +1.7 [-18.8, +22.2] |
