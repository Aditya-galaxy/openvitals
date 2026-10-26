# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| ens_hr | 1 | 0.507 [0.463, 0.549] | – | 66.6% [63.8%, 69.3%] | 0.635 | 53.7% | 58.2% | 78.2% | 85.7% | 0.492 ± 0.139 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| ens_hr | -1.4 [-40.3, +37.5] | -3.8 [-35.3, +27.7] | -1.1% [-14.1%, +12.0%] | +6.7 [-11.3, +24.7] |
