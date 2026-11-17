DREAMT held-out test: 100 subjects (lab PSG, Empatica E4 heart rate). Neither model saw DREAMT. Kappa CI = 95% subject bootstrap.

### `release` (ens10_hr, 4class)

| heart rate | subjects | kappa [95% CI] | median subject κ | accuracy | wake sens | light sens | deep sens | rem sens | TST bias, min [95% LoA] |
|---|---|---|---|---|---|---|---|---|---|
| 1 Hz (E4) | 100 | 0.235 [0.205, 0.267] | 0.223 | 46.1% | 44% | 43% | 76% | 59% | +23 [-97, +142] |
| 1 / 5 s | 100 | 0.236 [0.204, 0.267] | 0.224 | 46.1% | 44% | 43% | 76% | 59% | +23 [-97, +143] |
| 1 / 60 s | 100 | 0.226 [0.195, 0.256] | 0.211 | 45.5% | 43% | 43% | 75% | 56% | +23 [-98, +143] |

By sleep-apnea severity (1 / 5 s):

| group | subjects | kappa [95% CI] |
|---|---|---|
| AHI < 5 | 26 | 0.269 [0.209, 0.329] |
| AHI 5–15 | 25 | 0.257 [0.185, 0.330] |
| AHI 15–30 | 24 | 0.246 [0.194, 0.293] |
| AHI ≥ 30 | 25 | 0.168 [0.118, 0.222] |

### `val` (ens10_hr, 4class)

| heart rate | subjects | kappa [95% CI] | median subject κ | accuracy | wake sens | light sens | deep sens | rem sens | TST bias, min [95% LoA] |
|---|---|---|---|---|---|---|---|---|---|
| 1 Hz (E4) | 100 | 0.223 [0.193, 0.253] | 0.213 | 44.3% | 41% | 41% | 75% | 62% | +29 [-88, +146] |
| 1 / 5 s | 100 | 0.223 [0.193, 0.253] | 0.214 | 44.2% | 41% | 41% | 75% | 62% | +29 [-88, +146] |
| 1 / 60 s | 100 | 0.211 [0.181, 0.240] | 0.204 | 43.2% | 41% | 40% | 75% | 59% | +30 [-87, +147] |

By sleep-apnea severity (1 / 5 s):

| group | subjects | kappa [95% CI] |
|---|---|---|
| AHI < 5 | 26 | 0.250 [0.196, 0.306] |
| AHI 5–15 | 25 | 0.241 [0.172, 0.312] |
| AHI 15–30 | 24 | 0.238 [0.186, 0.284] |
| AHI ≥ 30 | 25 | 0.162 [0.112, 0.217] |
