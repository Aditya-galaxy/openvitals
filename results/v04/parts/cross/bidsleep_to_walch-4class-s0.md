# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 47 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.290 [0.260, 0.318] | – | 48.9% [46.8%, 50.9%] | 0.478 | 45.3% | 36.2% | 66.0% | 71.8% | 0.282 ± 0.077 |
| gbdt_motion_hr | 1 | 0.324 [0.293, 0.354] | – | 53.6% [51.2%, 55.6%] | 0.518 | 45.1% | 46.2% | 65.1% | 68.2% | 0.314 ± 0.087 |
| tcn_hr | 1 | 0.443 [0.399, 0.483] | – | 61.9% [58.9%, 64.5%] | 0.584 | 48.4% | 54.4% | 76.9% | 76.8% | 0.426 ± 0.131 |
| tcn_motion_hr | 1 | 0.467 [0.418, 0.513] | – | 64.1% [60.8%, 67.3%] | 0.602 | 44.0% | 58.2% | 80.6% | 77.0% | 0.452 ± 0.138 |
| gru_hr | 1 | 0.459 [0.417, 0.499] | – | 63.0% [60.0%, 65.8%] | 0.596 | 44.4% | 55.0% | 79.9% | 80.3% | 0.440 ± 0.137 |
| gru_motion_hr | 1 | 0.443 [0.396, 0.490] | – | 62.7% [59.8%, 65.6%] | 0.597 | 46.7% | 58.5% | 82.6% | 67.2% | 0.422 ± 0.140 |

## Paired comparison against `tcn_motion_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| gru_hr | -0.008 | [-0.031, +0.016] | 0.745 |
| gru_motion_hr | -0.024 | [-0.050, +0.001] | 0.966 |
| tcn_hr | -0.024 | [-0.046, -0.003] | 0.989 |
| gbdt_motion_hr | -0.143 | [-0.173, -0.112] | 1.000 |
| gbdt_hr | -0.177 | [-0.209, -0.144] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -2.8 [-48.4, +42.8] | +4.3 [-33.6, +42.2] | -1.1% [-13.8%, +11.7%] | +1.3 [-17.7, +20.3] |
| gbdt_motion_hr | +5.2 [-30.3, +40.8] | -1.4 [-33.1, +30.4] | +1.2% [-9.3%, +11.6%] | -1.1 [-21.8, +19.5] |
| tcn_hr | -4.0 [-48.6, +40.6] | +1.1 [-32.6, +34.8] | -1.6% [-15.0%, +11.9%] | +4.8 [-19.8, +29.3] |
| tcn_motion_hr | +4.5 [-38.1, +47.2] | -1.3 [-36.4, +33.7] | +0.6% [-11.8%, +12.9%] | -0.9 [-21.3, +19.4] |
| gru_hr | +6.0 [-31.8, +43.8] | -5.7 [-30.8, +19.4] | +1.0% [-9.8%, +11.8%] | +2.3 [-20.2, +24.8] |
| gru_motion_hr | +7.9 [-27.7, +43.4] | -5.7 [-32.3, +20.9] | +2.0% [-6.5%, +10.4%] | +0.3 [-21.5, +22.2] |
