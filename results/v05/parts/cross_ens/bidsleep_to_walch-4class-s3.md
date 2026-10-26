# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| ens_hr | 1 | 0.506 [0.468, 0.543] | – | 65.9% [63.2%, 68.5%] | 0.632 | 53.5% | 55.1% | 81.5% | 88.3% | 0.492 ± 0.123 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| ens_hr | +0.0 [-37.3, +37.4] | -4.3 [-35.0, +26.3] | -0.4% [-10.9%, +10.1%] | +5.1 [-13.5, +23.7] |
