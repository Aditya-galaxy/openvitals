# OpenVitals benchmark: walch_to_bidsleep / 4class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 253 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.298 [0.278, 0.318] | – | 50.8% [49.4%, 52.1%] | 0.494 | 53.3% | 52.8% | 52.6% | 45.0% | 0.293 ± 0.071 |
| tcn_hr | 1 | 0.433 [0.408, 0.455] | – | 60.3% [58.5%, 61.9%] | 0.583 | 55.4% | 61.4% | 66.1% | 56.2% | 0.428 ± 0.089 |
| gru_hr | 1 | 0.392 [0.364, 0.418] | – | 56.5% [54.5%, 58.3%] | 0.549 | 55.4% | 53.6% | 73.6% | 49.4% | 0.388 ± 0.098 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | -0.041 | [-0.051, -0.031] | 1.000 |
| gbdt_hr | -0.134 | [-0.148, -0.121] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -13.5 [-76.4, +49.3] | +17.1 [-24.5, +58.6] | -3.5% [-18.8%, +11.9%] | -1.8 [-45.2, +41.6] |
| tcn_hr | -15.5 [-77.7, +46.6] | +19.1 [-23.9, +62.0] | -3.9% [-18.5%, +10.8%] | -3.0 [-53.6, +47.5] |
| gru_hr | -19.9 [-89.7, +49.9] | +30.3 [-20.1, +80.8] | -4.9% [-21.6%, +11.7%] | -10.6 [-65.5, +44.4] |
