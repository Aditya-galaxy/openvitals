# OpenVitals benchmark: bidsleep_to_walch / 4class

- Data: bidsleep (47 subjects, 47 nights); test: walch (31 subjects, 31 nights)
- Folds: 1 (subject-wise; no subject in both train and test)
- Seeds per model: 5; CIs: 95% subject bootstrap (2000 draws)
- openvitals 0.1.0, Python 3.11.2

## Epoch agreement (pooled over test epochs and seeds)

| model | seeds | kappa [95% CI] | seed SD | accuracy [95% CI] | macro F1 | wake sens | light sens | deep sens | rem sens | subject kappa (mean ± sd) |
|---|---|---|---|---|---|---|---|---|---|---|
| gbdt_hr | 5 | 0.291 [0.261, 0.319] | 0.002 | 49.0% [46.9%, 51.1%] | 0.479 | 45.3% | 36.5% | 65.6% | 71.8% | 0.283 ± 0.077 |
| gbdt_motion_hr | 5 | 0.324 [0.294, 0.352] | 0.002 | 53.3% [51.1%, 55.3%] | 0.518 | 45.6% | 45.4% | 65.1% | 69.1% | 0.312 ± 0.085 |
| tcn_hr | 5 | 0.434 [0.390, 0.475] | 0.017 | 60.7% [57.8%, 63.4%] | 0.577 | 45.9% | 50.7% | 78.5% | 80.8% | 0.421 ± 0.125 |
| tcn_motion_hr | 5 | 0.447 [0.404, 0.487] | 0.013 | 61.9% [58.9%, 64.6%] | 0.588 | 43.8% | 52.7% | 81.9% | 79.7% | 0.431 ± 0.124 |
| gru_hr | 5 | 0.420 [0.378, 0.458] | 0.032 | 59.4% [56.4%, 61.9%] | 0.568 | 46.3% | 48.2% | 79.6% | 80.2% | 0.405 ± 0.123 |
| gru_motion_hr | 5 | 0.425 [0.381, 0.465] | 0.012 | 60.0% [57.0%, 62.7%] | 0.581 | 48.8% | 49.3% | 81.1% | 78.3% | 0.407 ± 0.124 |

## Paired comparison against `tcn_hr` (same subject draws)

| model | Δ kappa | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| tcn_motion_hr | +0.013 | [-0.003, +0.027] | 0.057 |
| gru_motion_hr | -0.009 | [-0.027, +0.009] | 0.843 |
| gru_hr | -0.014 | [-0.029, +0.000] | 0.974 |
| gbdt_motion_hr | -0.110 | [-0.133, -0.088] | 1.000 |
| gbdt_hr | -0.143 | [-0.165, -0.120] | 1.000 |

## Night-level agreement (predicted − PSG; bias [95% limits of agreement], averaged over seeds)

| model | TST min | WASO min | sleep efficiency | SOL min |
|---|---|---|---|---|
| gbdt_hr | -2.2 [-48.0, +43.7] | +4.0 [-34.3, +42.4] | -0.9% [-13.5%, +11.6%] | +1.0 [-18.1, +20.1] |
| gbdt_motion_hr | +5.0 [-30.3, +40.4] | -1.3 [-32.9, +30.2] | +1.1% [-9.4%, +11.6%] | -1.0 [-21.9, +19.9] |
| tcn_hr | +0.9 [-42.8, +44.6] | -1.2 [-33.4, +31.0] | -0.2% [-12.2%, +11.7%] | +2.6 [-19.6, +24.9] |
| tcn_motion_hr | +6.9 [-33.4, +47.3] | -3.4 [-33.8, +27.1] | +1.5% [-8.8%, +11.8%] | -1.2 [-21.5, +19.0] |
| gru_hr | +1.0 [-42.2, +44.2] | -2.0 [-35.4, +31.4] | -0.2% [-11.8%, +11.5%] | +3.2 [-21.7, +28.1] |
| gru_motion_hr | +4.4 [-34.6, +43.5] | -4.5 [-35.1, +26.0] | +0.9% [-9.3%, +11.0%] | +2.3 [-20.2, +24.9] |
