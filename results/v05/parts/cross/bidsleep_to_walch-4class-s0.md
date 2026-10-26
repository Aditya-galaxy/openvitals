# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.314 [0.279, 0.347] | – | 49.7% [46.9%, 52.2%] | 0.489 | 50.0% | 34.3% | 68.8% | 76.9% | 0.305 ± 0.092 |
| tcn_hr | 1 | 0.455 [0.410, 0.498] | – | 61.4% [58.0%, 64.5%] | 0.590 | 47.6% | 47.4% | 81.9% | 89.5% | 0.446 ± 0.132 |
| gru_hr | 1 | 0.483 [0.440, 0.525] | – | 63.5% [60.6%, 66.3%] | 0.615 | 60.2% | 49.8% | 83.1% | 87.2% | 0.467 ± 0.135 |

## Paired comparison against `gbdt_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | +0.169 | [+0.141, +0.197] | 0.000 |
| tcn_hr | +0.141 | [+0.116, +0.169] | 0.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -7.7 [-51.3, +35.9] | +7.4 [-29.0, +43.9] | -2.1% [-13.2%, +9.0%] | +3.2 [-16.2, +22.5] |
| tcn_hr | +2.4 [-34.0, +38.9] | -4.0 [-33.1, +25.2] | +0.3% [-8.2%, +8.8%] | +4.2 [-12.3, +20.8] |
| gru_hr | -12.2 [-56.3, +31.8] | +4.2 [-25.5, +33.8] | -3.3% [-15.2%, +8.6%] | +7.8 [-16.3, +31.9] |
