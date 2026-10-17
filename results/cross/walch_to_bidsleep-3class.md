# OpenVitals benchmark: walch_to_bidsleep / 3class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 253 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over all test epochs)

| model | accuracy | kappa | macro F1 | wake sens | nrem sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|
| majority | 63.1% | 0.000 | 0.258 | 0.0% | 100.0% | 0.0% | 0.000 ± 0.000 |
| logreg_hr | 58.7% | 0.281 | 0.503 | 61.7% | 67.3% | 36.9% | 0.279 ± 0.080 |
| gbdt_hr | 64.4% | 0.341 | 0.566 | 54.0% | 72.5% | 48.9% | 0.339 ± 0.092 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement])

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| majority | +44.7 [-12.7, +102.2] | -17.9 [-50.9, +15.1] | +10.8% [-2.8%, +24.4%] | -24.6 [-69.3, +20.1] |
| logreg_hr | -59.7 [-144.0, +24.5] | +74.3 [+8.7, +140.0] | -15.0% [-36.7%, +6.7%] | -12.7 [-56.7, +31.4] |
| gbdt_hr | -14.7 [-76.5, +47.1] | +17.9 [-22.4, +58.2] | -3.7% [-18.8%, +11.3%] | -1.4 [-43.9, +41.2] |
