# OpenVitals benchmark: walch_to_bidsleep / 4class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 253 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 5; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 5 | 0.301 [0.280, 0.320] | 0.002 | 51.0% [49.5%, 52.3%] | 0.496 | 53.4% | 52.8% | 52.5% | 45.7% | 0.296 ± 0.072 |
| tcn_hr | 5 | 0.434 [0.409, 0.458] | 0.005 | 60.2% [58.5%, 61.8%] | 0.584 | 53.3% | 58.6% | 68.2% | 59.9% | 0.430 ± 0.090 |
| gru_hr | 5 | 0.404 [0.375, 0.430] | 0.013 | 58.1% [56.1%, 59.9%] | 0.564 | 56.3% | 58.5% | 62.3% | 55.1% | 0.399 ± 0.095 |

## Paired comparison against `gbdt_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | +0.133 | [+0.120, +0.147] | 0.000 |
| gru_hr | +0.102 | [+0.088, +0.118] | 0.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -13.0 [-74.6, +48.6] | +16.3 [-23.6, +56.3] | -3.3% [-18.3%, +11.7%] | -1.6 [-44.5, +41.4] |
| tcn_hr | -11.4 [-73.7, +50.9] | +17.9 [-23.5, +59.2] | -2.8% [-17.5%, +11.9%] | -6.2 [-56.4, +44.0] |
| gru_hr | -15.8 [-79.5, +48.0] | +21.5 [-25.8, +68.8] | -3.9% [-19.1%, +11.4%] | -5.6 [-58.4, +47.2] |
