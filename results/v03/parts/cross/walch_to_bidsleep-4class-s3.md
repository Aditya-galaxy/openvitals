# OpenVitals benchmark: walch_to_bidsleep / 4class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 253 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.303 [0.282, 0.322] | – | 51.0% [49.6%, 52.3%] | 0.497 | 53.7% | 52.5% | 52.7% | 46.2% | 0.298 ± 0.072 |
| tcn_hr | 1 | 0.438 [0.411, 0.462] | – | 60.2% [58.4%, 61.9%] | 0.586 | 57.2% | 58.6% | 69.1% | 57.8% | 0.433 ± 0.092 |
| gru_hr | 1 | 0.427 [0.394, 0.457] | – | 59.5% [57.1%, 61.6%] | 0.583 | 58.8% | 56.4% | 63.0% | 62.4% | 0.423 ± 0.103 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | -0.011 | [-0.024, +0.001] | 0.963 |
| gbdt_hr | -0.135 | [-0.149, -0.121] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -13.5 [-74.6, +47.5] | +16.8 [-22.3, +56.0] | -3.4% [-18.3%, +11.4%] | -1.6 [-44.4, +41.3] |
| tcn_hr | -18.4 [-83.7, +46.8] | +22.4 [-20.8, +65.6] | -4.5% [-19.8%, +10.8%] | -5.1 [-53.1, +42.8] |
| gru_hr | -12.6 [-73.2, +48.1] | +13.1 [-33.2, +59.3] | -3.1% [-18.0%, +11.8%] | +0.6 [-55.3, +56.5] |
