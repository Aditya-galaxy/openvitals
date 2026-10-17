# OpenVitals benchmark: walch_to_bidsleep / 4class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 253 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.300 [0.279, 0.319] | – | 50.8% [49.4%, 52.2%] | 0.495 | 53.1% | 52.5% | 52.5% | 45.8% | 0.294 ± 0.073 |
| tcn_hr | 1 | 0.438 [0.414, 0.459] | – | 60.7% [59.0%, 62.2%] | 0.588 | 54.0% | 59.8% | 61.2% | 64.4% | 0.433 ± 0.085 |
| gru_hr | 1 | 0.403 [0.373, 0.431] | – | 57.8% [55.7%, 59.7%] | 0.561 | 56.6% | 56.9% | 65.2% | 54.4% | 0.400 ± 0.099 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | -0.034 | [-0.047, -0.022] | 1.000 |
| gbdt_hr | -0.138 | [-0.152, -0.124] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -12.7 [-74.3, +48.9] | +16.0 [-23.6, +55.7] | -3.2% [-18.3%, +11.8%] | -1.6 [-44.5, +41.3] |
| tcn_hr | -11.3 [-70.3, +47.6] | +17.6 [-21.0, +56.1] | -2.8% [-16.8%, +11.2%] | -5.7 [-55.2, +43.8] |
| gru_hr | -18.8 [-84.2, +46.5] | +24.3 [-22.5, +71.0] | -4.5% [-19.7%, +10.7%] | -6.1 [-56.8, +44.7] |
