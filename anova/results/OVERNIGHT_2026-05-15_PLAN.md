# Overnight run — 2026-05-15 (Anna stepping out 3-4h)

Five experiments queued in serial. Each writes its own JSON result file
so partial completions still produce paper-ready outputs.

## The big-bet experiment

**Architecturally-restricted LC-PINN training**: if our Sobol indices for
2D Helm say $S_k + S_{xk} + S_{yk} + S_{xyk} = 0.64$ (training prior), then a
PINN whose *architecture* is restricted to be additive — $u(x,y,k) =
f_{xy}(x,y) + f_k(k)$ — can capture at most $S_x + S_y + S_{xy} + S_k =
0.36$ of variance. So its rel-$L^2$ vs analytic reference must satisfy

$$\mathrm{rel}\text{-}L^2 \ge \sqrt{S_{xk} + S_{yk} + S_{xyk}} = \sqrt{0.58} \approx 0.76.$$

Similarly an order-2 PINN (additive + pair terms, no triplet) has floor
$\sqrt{S_{xyk}} \approx 0.66$.

If we train these architectures and they hit these floors, **we've shown
Sobol indices give an a priori architectural complexity bound on the
network class needed to solve a parametric PDE.** This is the strongest
single new claim we could add to the paper.

## Run order and budget

| # | Experiment | Est runtime | Output |
|---|---|---|---|
| 1 | Synthetic-truth validation | 5 min | `results/synthetic_truth.json` |
| 2 | HDMR truncation post-hoc rel-$L^2$ | 15 min | `results/hdmr_truncation_floors.json` |
| 3 | 3D Helm order-3 full decomposition | 30 min | `results/helm3d_order3_full.json` |
| 4 | Method ablation (Fourier / purification / phase 2) | 30 min | `results/method_ablation.json` |
| 5 | Architecturally-restricted LC-PINN training | 90 min | `results/restricted_lcpinn.json` |

Total: ~3 hours.

## Status log

(updated by each experiment as it completes)

## Run log (2026-05-15 16:07)

- 1/5 Synthetic-truth validation                     DONE       2341s
- 2/5 HDMR truncation floors (post-hoc)              DONE       735s
- 3/5 3D Helm order-3 full decomposition             DONE       1s
- 4/5 Method ablation                                DONE       1117s
- 5/5 Architecturally-restricted LC-PINN             DONE       3131s

## Run complete at Fri May 15 18:09:23 PDT 2026

