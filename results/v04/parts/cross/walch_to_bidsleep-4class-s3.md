# OpenVitals benchmark: walch_to_bidsleep / 4class

- Data: walch (31 subjects, 31 nights); test: bidsleep (47 subjects, 47 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 1; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 1 | 0.313 [0.276, 0.346] | – | 51.8% [49.3%, 54.1%] | 0.508 | 56.7% | 51.0% | 56.2% | 48.2% | 0.304 ± 0.126 |
| gbdt_motion_hr | 1 | 0.300 [0.267, 0.330] | – | 52.4% [50.0%, 54.4%] | 0.497 | 65.0% | 62.0% | 41.6% | 38.0% | 0.293 ± 0.112 |
| tcn_hr | 1 | 0.406 [0.354, 0.452] | – | 61.1% [57.7%, 64.0%] | 0.562 | 28.5% | 71.2% | 51.3% | 63.0% | 0.399 ± 0.174 |
| tcn_motion_hr | 1 | 0.412 [0.362, 0.455] | – | 60.6% [57.4%, 63.5%] | 0.579 | 58.7% | 70.2% | 49.1% | 52.8% | 0.407 ± 0.163 |
| gru_hr | 1 | 0.443 [0.395, 0.489] | – | 60.9% [57.5%, 64.1%] | 0.601 | 63.4% | 56.4% | 62.7% | 66.5% | 0.440 ± 0.168 |
| gru_motion_hr | 1 | 0.367 [0.313, 0.417] | – | 54.2% [49.8%, 58.1%] | 0.533 | 78.6% | 53.1% | 57.3% | 44.4% | 0.367 ± 0.169 |

## Paired comparison against `gru_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | -0.031 | [-0.069, +0.005] | 0.956 |
| tcn_hr | -0.037 | [-0.063, -0.013] | 0.999 |
| gru_motion_hr | -0.075 | [-0.113, -0.040] | 1.000 |
| gbdt_hr | -0.130 | [-0.160, -0.099] | 1.000 |
| gbdt_motion_hr | -0.143 | [-0.179, -0.110] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -17.4 [-64.9, +30.2] | +18.0 [-19.9, +55.9] | -4.1% [-15.3%, +7.1%] | +0.2 [-32.8, +33.2] |
| gbdt_motion_hr | -30.2 [-83.5, +23.1] | +27.9 [-18.1, +73.8] | -6.8% [-17.9%, +4.2%] | +1.3 [-29.2, +31.8] |
| tcn_hr | +24.5 [-26.9, +75.8] | -4.9 [-45.3, +35.5] | +5.7% [-6.1%, +17.4%] | -18.6 [-57.0, +19.8] |
| tcn_motion_hr | -15.1 [-84.5, +54.3] | +18.5 [-44.3, +81.4] | -3.2% [-18.1%, +11.8%] | -6.6 [-43.2, +30.0] |
| gru_hr | -12.8 [-54.2, +28.7] | +8.8 [-38.9, +56.4] | -2.9% [-12.0%, +6.1%] | +3.6 [-43.3, +50.6] |
| gru_motion_hr | -76.9 [-200.2, +46.3] | +58.6 [-51.6, +168.8] | -17.4% [-42.3%, +7.6%] | +9.6 [-21.5, +40.6] |
