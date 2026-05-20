# Master summary of LC-PINN × ANOVA experiments

_Auto-generated from `results/` JSON files._


## 1D Helmholtz — Fourier joint HDMR (d=2)

| seed-tag | LC-PINN rel-L² | HDMR val rel-RMSE | S_x | S_k | S_{x,k} |
|---|---|---|---|---|---|
| helm1d_seed0 | 0.0007 | 0.0357 | 0.141 | 0.205 | 0.654 |
| helm1d_seed1 | 0.0009 | 0.0429 | 0.147 | 0.211 | 0.643 |
| helm1d_seed2 | 0.0012 | 0.0379 | 0.146 | 0.201 | 0.653 |
| helm1d_seed3 | 0.0016 | 0.0452 | 0.142 | 0.203 | 0.655 |

**Cross-pair Sobol $S_{x, k}$:** mean=0.6514  std=0.0050  (across 4 seeds)

## Schrödinger 1D — Fourier joint HDMR (d=2)

| seed-tag | LC-PINN rel-L² | HDMR val rel-RMSE | S_x | S_α | S_{x,α} |
|---|---|---|---|---|---|
| schr1d_seed0 | 0.0001 | 0.0136 | 0.986 | 0.010 | 0.004 |
| schr1d_seed1 | 0.0001 | 0.0117 | 0.987 | 0.009 | 0.004 |
| schr1d_seed2 | 0.0002 | 0.0103 | 0.986 | 0.010 | 0.004 |
| schr1d_seed3 | 0.0001 | 0.0109 | 0.986 | 0.010 | 0.004 |

**Spatial main $S_x$:** mean=0.9863  std=0.0005
**Cross-pair $S_{x, \alpha}$:** mean=0.0039  std=0.0001  (across 4 seeds)

## 2D Helmholtz — Fourier joint HDMR (d=3, order-3)

| tag | val rel-RMSE | S_x | S_y | S_k | S_{x,y} | S_{x,k} | S_{y,k} | **S_{x,y,k}** |
|---|---|---|---|---|---|---|---|---|
| fourier_h128_L6_p3eps400_seed0 | 0.4526 | 0.025 | 0.033 | 0.033 | 0.136 | 0.118 | 0.087 | 0.567 ||
| helm2d_fourier_seed2 | 0.5369 | 0.064 | 0.018 | 0.029 | 0.135 | 0.097 | 0.076 | 0.582 ||
| helm2d_fourier_seed3 | 0.3245 | 0.039 | 0.042 | 0.039 | 0.209 | 0.082 | 0.064 | 0.525 ||

**Triplet $S_{x,y,k}$ (HDMR-normalised):** mean=0.5583  std=0.0240

## 2D Helmholtz — MC-Sobol (gold standard, d=3)

| seed-tag | S_x | S_y | S_k | S_{x,y} | S_{x,k} | S_{y,k} | **S_{x,y,k}** |
|---|---|---|---|---|---|---|---|
| mc_sobol_seed0 | 0.044 | 0.039 | 0.076 | 0.258 | 0.064 | 0.078 | 0.441 |
| mc_sobol_seed1 | 0.043 | 0.039 | 0.077 | 0.259 | 0.066 | 0.077 | 0.439 |
| mc_sobol_seed2 | 0.044 | 0.039 | 0.076 | 0.258 | 0.063 | 0.077 | 0.441 |
| mc_sobol_seed3 | 0.044 | 0.039 | 0.076 | 0.258 | 0.064 | 0.078 | 0.442 |

**MC triplet $S_{x,y,k}$:** mean=0.4408  std=0.0011
