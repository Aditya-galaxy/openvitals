# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| ens_hr | 1 | 0.478 [0.434, 0.522] | – | 63.1% [60.0%, 66.1%] | 0.610 | 54.2% | 48.8% | 83.3% | 90.1% | 0.466 ± 0.132 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| ens_hr | -3.4 [-40.8, +34.1] | -1.1 [-32.3, +30.2] | -1.3% [-11.4%, +8.7%] | +5.8 [-12.1, +23.7] |
