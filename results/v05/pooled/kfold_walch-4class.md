# OpenVitals benchmark: kfold_walch / 4class

- Data: bidsleep (47 subjects, 253 nights), walch (31 subjects, 31 nights); scored on walch only
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 5; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 5 | 0.319 [0.285, 0.352] | 0.001 | 50.4% [47.7%, 52.8%] | 0.494 | 49.2% | 35.5% | 68.1% | 77.5% | 0.309 ± 0.091 |
| tcn_hr | 5 | 0.488 [0.442, 0.530] | 0.002 | 64.6% [61.7%, 67.4%] | 0.619 | 52.9% | 54.3% | 84.2% | 83.1% | 0.472 ± 0.138 |
| gru_hr | 5 | 0.497 [0.458, 0.535] | 0.004 | 65.5% [63.0%, 68.1%] | 0.628 | 57.9% | 56.8% | 82.7% | 80.0% | 0.479 ± 0.132 |

## Paired comparison against `gru_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.009 | [-0.025, +0.007] | 0.879 |
| gbdt_hr | -0.178 | [-0.204, -0.153] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -6.1 [-50.9, +38.6] | +6.0 [-30.5, +42.5] | -1.8% [-13.4%, +9.8%] | +3.0 [-16.0, +22.0] |
| tcn_hr | -1.6 [-44.9, +41.8] | -1.7 [-36.5, +33.0] | -0.8% [-12.4%, +10.7%] | +4.2 [-14.5, +22.9] |
| gru_hr | -8.3 [-59.8, +43.2] | +2.2 [-42.3, +46.6] | -2.7% [-18.3%, +13.0%] | +7.0 [-16.9, +30.8] |
