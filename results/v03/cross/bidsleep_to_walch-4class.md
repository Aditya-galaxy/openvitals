# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 5; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 5 | 0.313 [0.279, 0.346] | 0.001 | 49.7% [47.0%, 52.2%] | 0.488 | 49.6% | 34.5% | 68.1% | 77.0% | 0.304 ± 0.088 |
| tcn_hr | 5 | 0.482 [0.440, 0.523] | 0.016 | 64.2% [61.4%, 66.9%] | 0.613 | 51.5% | 53.6% | 81.3% | 85.2% | 0.469 ± 0.133 |
| gru_hr | 5 | 0.488 [0.449, 0.526] | 0.009 | 64.7% [62.2%, 67.2%] | 0.621 | 55.1% | 54.7% | 80.4% | 84.1% | 0.473 ± 0.128 |

## Paired comparison against `gbdt_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | +0.175 | [+0.149, +0.202] | 0.000 |
| tcn_hr | +0.169 | [+0.144, +0.197] | 0.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -7.3 [-51.3, +36.6] | +7.0 [-29.8, +43.7] | -2.0% [-13.2%, +9.1%] | +3.3 [-16.3, +22.8] |
| tcn_hr | -1.5 [-42.0, +39.1] | -0.9 [-34.0, +32.2] | -0.7% [-11.1%, +9.7%] | +3.7 [-15.8, +23.2] |
| gru_hr | -4.9 [-49.2, +39.3] | -1.2 [-34.5, +32.1] | -1.6% [-14.7%, +11.4%] | +6.4 [-15.7, +28.5] |
