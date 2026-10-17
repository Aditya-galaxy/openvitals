# OpenVitals benchmark: walch_to_bidsleep / 4class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 253 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.303 [0.282, 0.322] | – | 51.1% [49.7%, 52.5%] | 0.498 | 53.5% | 53.2% | 52.1% | 45.8% | 0.298 ± 0.071 |
| tcn_hr | 1 | 0.436 [0.410, 0.461] | – | 60.7% [58.9%, 62.5%] | 0.587 | 47.9% | 60.6% | 70.5% | 59.1% | 0.432 ± 0.099 |
| gru_hr | 1 | 0.397 [0.368, 0.423] | – | 58.8% [56.9%, 60.6%] | 0.559 | 50.9% | 65.5% | 48.3% | 58.4% | 0.390 ± 0.096 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | -0.039 | [-0.056, -0.024] | 1.000 |
| gbdt_hr | -0.134 | [-0.150, -0.118] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -12.7 [-74.3, +48.9] | +16.0 [-23.6, +55.6] | -3.3% [-18.3%, +11.7%] | -1.5 [-44.5, +41.4] |
| tcn_hr | -1.0 [-63.2, +61.2] | +11.2 [-28.5, +51.0] | -0.3% [-14.9%, +14.4%] | -9.9 [-60.0, +40.3] |
| gru_hr | -9.0 [-70.9, +53.0] | +18.2 [-26.9, +63.2] | -2.2% [-17.2%, +12.7%] | -9.3 [-58.1, +39.4] |
