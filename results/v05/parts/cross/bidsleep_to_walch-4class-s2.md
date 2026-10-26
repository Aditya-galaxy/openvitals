# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.313 [0.279, 0.345] | – | 49.7% [47.0%, 52.2%] | 0.489 | 49.8% | 34.5% | 68.0% | 76.8% | 0.303 ± 0.088 |
| tcn_hr | 1 | 0.486 [0.442, 0.529] | – | 64.6% [61.8%, 67.4%] | 0.621 | 54.5% | 54.6% | 80.1% | 84.5% | 0.473 ± 0.135 |
| gru_hr | 1 | 0.504 [0.463, 0.545] | – | 66.6% [63.9%, 69.4%] | 0.632 | 52.5% | 59.4% | 76.4% | 84.5% | 0.490 ± 0.140 |

## Paired comparison against `gbdt_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | +0.191 | [+0.159, +0.223] | 0.000 |
| tcn_hr | +0.173 | [+0.145, +0.204] | 0.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -7.6 [-51.7, +36.5] | +7.1 [-30.0, +44.2] | -2.0% [-13.0%, +9.0%] | +3.4 [-16.3, +23.0] |
| tcn_hr | -2.7 [-39.2, +33.8] | -0.4 [-31.0, +30.2] | -0.9% [-9.7%, +7.9%] | +4.7 [-14.2, +23.6] |
| gru_hr | -2.5 [-44.3, +39.3] | -4.6 [-34.6, +25.3] | -1.3% [-15.3%, +12.6%] | +8.7 [-13.2, +30.7] |
