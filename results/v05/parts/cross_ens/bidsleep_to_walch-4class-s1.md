# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| ens_hr | 1 | 0.505 [0.462, 0.548] | – | 66.2% [63.4%, 68.9%] | 0.633 | 56.4% | 57.1% | 78.1% | 85.8% | 0.488 ± 0.143 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| ens_hr | -5.6 [-50.4, +39.2] | +0.0 [-36.5, +36.5] | -2.1% [-17.0%, +12.9%] | +6.3 [-16.7, +29.2] |
