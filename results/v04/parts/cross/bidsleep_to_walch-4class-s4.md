# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 47 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.288 [0.258, 0.316] | – | 48.9% [46.8%, 51.0%] | 0.478 | 44.8% | 36.8% | 65.1% | 71.2% | 0.280 ± 0.079 |
| gbdt_motion_hr | 1 | 0.324 [0.295, 0.350] | – | 53.1% [51.0%, 55.0%] | 0.517 | 45.9% | 44.6% | 65.7% | 69.7% | 0.312 ± 0.083 |
| tcn_hr | 1 | 0.441 [0.394, 0.485] | – | 60.8% [57.7%, 63.7%] | 0.583 | 45.7% | 48.8% | 79.1% | 85.7% | 0.430 ± 0.130 |
| tcn_motion_hr | 1 | 0.444 [0.397, 0.487] | – | 61.6% [58.3%, 64.4%] | 0.591 | 41.2% | 51.5% | 82.8% | 81.9% | 0.429 ± 0.139 |
| gru_hr | 1 | 0.397 [0.343, 0.443] | – | 57.5% [53.5%, 60.9%] | 0.556 | 41.5% | 44.1% | 77.7% | 85.1% | 0.391 ± 0.134 |
| gru_motion_hr | 1 | 0.426 [0.380, 0.469] | – | 59.7% [56.3%, 62.6%] | 0.583 | 52.4% | 47.4% | 79.8% | 81.4% | 0.408 ± 0.127 |

## Paired comparison against `tcn_motion_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_hr | -0.003 | [-0.023, +0.018] | 0.623 |
| gru_motion_hr | -0.018 | [-0.041, +0.002] | 0.965 |
| gru_hr | -0.048 | [-0.076, -0.019] | 0.999 |
| gbdt_motion_hr | -0.121 | [-0.148, -0.093] | 1.000 |
| gbdt_hr | -0.156 | [-0.188, -0.125] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -1.6 [-46.9, +43.6] | +3.6 [-34.1, +41.4] | -0.8% [-13.0%, +11.4%] | +0.8 [-17.4, +19.0] |
| gbdt_motion_hr | +5.1 [-30.5, +40.8] | -1.7 [-33.8, +30.4] | +1.1% [-9.6%, +11.7%] | -0.7 [-21.5, +20.0] |
| tcn_hr | +5.8 [-35.2, +46.9] | -6.3 [-37.4, +24.9] | +0.8% [-10.4%, +12.0%] | +3.1 [-18.4, +24.7] |
| tcn_motion_hr | +14.5 [-23.6, +52.7] | -10.0 [-37.9, +17.9] | +3.6% [-5.6%, +12.8%] | -2.4 [-21.8, +16.9] |
| gru_hr | +8.2 [-31.6, +48.0] | -8.0 [-39.3, +23.3] | +2.0% [-6.8%, +10.8%] | +2.2 [-19.9, +24.2] |
| gru_motion_hr | +0.1 [-43.5, +43.7] | -1.4 [-35.0, +32.1] | -0.0% [-10.6%, +10.6%] | +3.3 [-22.9, +29.5] |
