# OpenVitals benchmark: kfold / 4class

- Data: walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.309 [0.270, 0.346] | – | 53.3% [50.9%, 55.7%] | 0.494 | 48.6% | 51.3% | 52.2% | 61.1% | 0.305 ± 0.110 |
| gbdt_motion_hr | 1 | 0.368 [0.327, 0.408] | – | 60.1% [57.5%, 62.6%] | 0.550 | 59.4% | 65.2% | 44.8% | 57.0% | 0.355 ± 0.119 |
| tcn_hr | 1 | 0.489 [0.443, 0.535] | – | 66.1% [62.9%, 69.1%] | 0.621 | 58.0% | 62.6% | 71.3% | 74.8% | 0.479 ± 0.129 |
| tcn_motion_hr | 1 | 0.477 [0.419, 0.531] | – | 65.4% [61.7%, 68.9%] | 0.622 | 66.4% | 61.4% | 62.3% | 77.0% | 0.462 ± 0.164 |
| gru_hr | 1 | 0.427 [0.372, 0.476] | – | 60.9% [57.2%, 64.2%] | 0.579 | 55.8% | 54.1% | 72.0% | 72.9% | 0.419 ± 0.136 |
| gru_motion_hr | 1 | 0.457 [0.401, 0.506] | – | 63.4% [59.7%, 66.5%] | 0.612 | 69.9% | 57.7% | 69.9% | 71.1% | 0.444 ± 0.146 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | -0.012 | [-0.045, +0.023] | 0.775 |
| gru_motion_hr | -0.032 | [-0.071, +0.006] | 0.956 |
| gru_hr | -0.062 | [-0.093, -0.034] | 1.000 |
| gbdt_motion_hr | -0.122 | [-0.165, -0.078] | 1.000 |
| gbdt_hr | -0.181 | [-0.217, -0.146] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -11.0 [-63.6, +41.5] | +11.9 [-29.6, +53.4] | -2.6% [-15.5%, +10.3%] | +2.0 [-18.1, +22.0] |
| gbdt_motion_hr | -9.9 [-55.9, +36.2] | +11.2 [-28.9, +51.3] | -2.5% [-13.9%, +8.8%] | +1.3 [-19.8, +22.5] |
| tcn_hr | -13.6 [-65.4, +38.3] | +13.3 [-32.5, +59.1] | -3.3% [-15.7%, +9.0%] | +2.6 [-17.3, +22.5] |
| tcn_motion_hr | -14.2 [-65.2, +36.8] | +14.0 [-24.6, +52.6] | -3.7% [-16.9%, +9.4%] | +2.1 [-22.8, +27.0] |
| gru_hr | -11.3 [-68.0, +45.4] | +9.9 [-38.5, +58.3] | -2.4% [-15.3%, +10.5%] | +3.4 [-20.4, +27.3] |
| gru_motion_hr | -17.8 [-72.3, +36.7] | +16.2 [-25.6, +57.9] | -4.4% [-19.3%, +10.5%] | +3.4 [-19.2, +26.1] |
