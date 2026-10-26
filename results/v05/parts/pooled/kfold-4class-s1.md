# OpenVitals benchmark: kfold / 4class

- Data: bidsleep (47 subjects, 253 nights), walch (31 subjects, 31 nights)
- Folds: 5 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.361 [0.339, 0.380] | – | 53.7% [52.0%, 55.0%] | 0.538 | 54.4% | 40.8% | 68.1% | 65.8% | 0.342 ± 0.085 |
| tcn_hr | 1 | 0.507 [0.478, 0.531] | – | 65.2% [63.2%, 67.0%] | 0.642 | 58.8% | 58.4% | 77.8% | 70.9% | 0.490 ± 0.125 |
| gru_hr | 1 | 0.525 [0.501, 0.546] | – | 66.7% [65.0%, 68.3%] | 0.655 | 59.1% | 62.0% | 78.0% | 70.0% | 0.504 ± 0.121 |

## Paired comparison against `gbdt_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | +0.164 | [+0.150, +0.179] | 0.000 |
| tcn_hr | +0.146 | [+0.133, +0.161] | 0.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -3.5 [-60.8, +53.9] | +5.8 [-29.8, +41.3] | -1.0% [-15.0%, +12.9%] | -0.8 [-40.1, +38.5] |
| tcn_hr | -2.0 [-56.1, +52.1] | +1.6 [-36.4, +39.6] | -0.7% [-13.8%, +12.5%] | +0.7 [-40.6, +41.9] |
| gru_hr | -1.1 [-57.5, +55.3] | +0.2 [-38.1, +38.5] | -0.4% [-14.5%, +13.7%] | +0.4 [-37.3, +38.0] |
