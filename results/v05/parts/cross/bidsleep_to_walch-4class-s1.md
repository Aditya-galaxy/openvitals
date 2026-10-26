# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.313 [0.278, 0.346] | – | 49.7% [47.0%, 52.2%] | 0.488 | 49.2% | 34.7% | 68.0% | 76.5% | 0.303 ± 0.089 |
| tcn_hr | 1 | 0.498 [0.455, 0.539] | – | 65.6% [62.8%, 68.3%] | 0.624 | 56.4% | 56.2% | 77.7% | 85.4% | 0.485 ± 0.134 |
| gru_hr | 1 | 0.486 [0.443, 0.528] | – | 64.8% [62.0%, 67.4%] | 0.623 | 55.9% | 55.5% | 79.0% | 83.0% | 0.468 ± 0.148 |

## Paired comparison against `gbdt_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | +0.185 | [+0.159, +0.213] | 0.000 |
| gru_hr | +0.173 | [+0.146, +0.203] | 0.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -7.4 [-52.3, +37.5] | +7.0 [-30.5, +44.4] | -2.0% [-13.2%, +9.2%] | +3.4 [-16.3, +23.0] |
| tcn_hr | -9.8 [-58.9, +39.3] | +6.5 [-34.5, +47.6] | -2.5% [-14.0%, +9.0%] | +4.0 [-16.5, +24.4] |
| gru_hr | -4.5 [-46.2, +37.2] | -1.8 [-34.4, +30.8] | -1.9% [-17.1%, +13.3%] | +6.3 [-16.9, +29.6] |
