# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| ens10_hr | 1 | 0.514 [0.469, 0.558] | – | 66.6% [63.6%, 69.5%] | 0.638 | 54.0% | 56.1% | 82.7% | 87.9% | 0.498 ± 0.142 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| ens10_hr | -0.3 [-38.3, +37.7] | -3.6 [-36.6, +29.4] | -0.8% [-13.5%, +12.0%] | +5.2 [-12.9, +23.2] |
