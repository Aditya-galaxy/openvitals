# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 253 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.15

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.313 [0.280, 0.346] | – | 49.7% [47.0%, 52.2%] | 0.488 | 49.2% | 34.5% | 67.8% | 77.2% | 0.305 ± 0.087 |
| tcn_hr | 1 | 0.483 [0.441, 0.523] | – | 64.1% [61.3%, 66.8%] | 0.614 | 54.4% | 53.0% | 81.4% | 85.1% | 0.468 ± 0.137 |
| gru_hr | 1 | 0.483 [0.447, 0.519] | – | 64.4% [61.6%, 67.0%] | 0.618 | 51.9% | 53.9% | 78.6% | 87.0% | 0.475 ± 0.112 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | -0.000 | [-0.025, +0.025] | 0.518 |
| gbdt_hr | -0.170 | [-0.197, -0.146] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -7.4 [-51.0, +36.2] | +6.9 [-29.4, +43.1] | -2.0% [-13.1%, +9.0%] | +3.4 [-16.3, +23.0] |
| tcn_hr | -4.8 [-43.8, +34.3] | -0.0 [-33.6, +33.5] | -1.9% [-14.9%, +11.2%] | +4.7 [-15.5, +24.9] |
| gru_hr | +1.0 [-40.6, +42.6] | -3.9 [-35.9, +28.0] | +0.4% [-8.9%, +9.8%] | +4.4 [-18.2, +26.9] |
