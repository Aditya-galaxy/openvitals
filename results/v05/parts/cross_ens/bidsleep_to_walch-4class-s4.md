# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| ens_hr | 1 | 0.502 [0.453, 0.548] | – | 65.9% [62.7%, 68.9%] | 0.629 | 50.8% | 56.8% | 86.5% | 82.1% | 0.486 ± 0.149 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| ens_hr | +1.9 [-39.4, +43.2] | -4.8 [-38.8, +29.2] | +0.0% [-11.5%, +11.6%] | +2.4 [-14.9, +19.7] |
