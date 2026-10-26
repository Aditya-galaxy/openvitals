# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.313 [0.279, 0.345] | – | 49.7% [47.0%, 52.1%] | 0.489 | 49.5% | 34.3% | 68.1% | 77.3% | 0.303 ± 0.087 |
| tcn_hr | 1 | 0.489 [0.442, 0.534] | – | 65.2% [62.2%, 68.1%] | 0.617 | 44.3% | 57.0% | 85.5% | 81.7% | 0.474 ± 0.148 |
| gru_hr | 1 | 0.484 [0.442, 0.523] | – | 64.4% [61.6%, 67.0%] | 0.615 | 54.7% | 55.1% | 85.0% | 79.0% | 0.468 ± 0.133 |

## Paired comparison against `gbdt_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | +0.176 | [+0.144, +0.210] | 0.000 |
| gru_hr | +0.171 | [+0.143, +0.201] | 0.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -6.6 [-50.1, +36.9] | +6.4 [-29.9, +42.8] | -1.9% [-13.2%, +9.5%] | +3.0 [-16.5, +22.5] |
| tcn_hr | +7.6 [-34.1, +49.3] | -6.7 [-37.7, +24.3] | +1.6% [-8.6%, +11.8%] | +0.8 [-20.6, +22.2] |
| gru_hr | -6.4 [-58.5, +45.7] | +0.3 [-42.1, +42.6] | -2.1% [-16.8%, +12.6%] | +4.7 [-14.0, +23.3] |
