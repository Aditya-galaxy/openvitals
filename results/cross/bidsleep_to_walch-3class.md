# OpenVitals benchmark: bidsleep_to_walch / 3class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | nrem sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|
| majority | 69.5% | 0.000 | 0.273 | 0.0% | 100.0% | 0.0% | 0.000 ± 0.000 |
| logreg_hr | 58.9% | 0.305 | 0.506 | 52.2% | 57.6% | 65.5% | 0.291 ± 0.120 |
| gbdt_hr | 62.1% | 0.357 | 0.557 | 51.7% | 57.6% | 80.7% | 0.352 ± 0.109 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| majority | +36.2 [+1.5, +70.9] | -20.3 [-50.2, +9.7] | +9.4% [-1.2%, +19.9%] | -13.0 [-30.6, +4.7] |
| logreg_hr | -49.0 [-118.8, +20.9] | +52.6 [-3.0, +108.2] | -12.3% [-31.2%, +6.7%] | -0.7 [-32.8, +31.4] |
| gbdt_hr | -15.4 [-64.2, +33.3] | +13.2 [-26.4, +52.8] | -4.4% [-19.0%, +10.1%] | +5.1 [-17.5, +27.6] |
