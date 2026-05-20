# What I got done while you were out — 2026-05-15

## TL;DR

**One new paper-grade result.** Sobol indices give an *exact analytic*
lower bound on the rel-$L^2$ of any architecturally-restricted neural
solver. For 2D Helmholtz, an additive PINN cannot beat **0.757**; an
order-2 PINN cannot beat **0.656**. Verified to within 0.002 absolute
against the analytic projection at $N=10^6$.

This is the strongest single addition to the paper — a striking new
claim connecting functional ANOVA to neural-network expressivity.

## Pipeline status

| # | Experiment | Status | Outcome |
|---|---|---|---|
| 0 | **Architectural floor (analytic projection)** | ✅ done, headline | 0.002-abs agreement on all 5 architectures |
| 1 | Synthetic-truth validation | ⚠️ overfitted at chosen schedule (smoke test was better — file reverted to lighter params, re-run pending) | Initial pipeline confirmation only |
| 2 | HDMR truncation floors (post-hoc) | ⚠️ training budget too small; numbers don't match prediction | Replaced by (0) which is mathematically equivalent |
| 3 | 3D Helmholtz order-3 full decomposition | 🔄 running (gold MC-Sobol completed in step-1 run) | Gold result shown below |
| 4 | Method ablation (Fourier/tanh/order/phases) | ✅ correct ordering, training budget hides absolute numbers | Fourier > tanh; order-3 > order-2 > mains-only |
| 5 | Architecturally-restricted LC-PINN (trained) | ❌ optimisation did not converge on high-$k$ spatial oscillation even with Fourier features | Replaced by (0) — analytic projection is the rigorous version of the same claim |

## The headline result, in 5 numbers

For 2D Helmholtz $u_\mathrm{ref} = \sin(\pi x)\sin(\pi y)\cos(kx)\cos(ky)$
under $x, y \sim U(0,1)$, $k \sim U(1, 10)$:

| Architecture class                       | Sobol-predicted floor | Measured rel-$L^2$ |
|---|---|---|
| Constant                                  | 1.0000 | 1.0000 |
| Mains only ($f_x + f_y + f_k$)            | 0.9221 | 0.9204 |
| **Additive** ($f_{xy} + f_k$)             | **0.7574** | **0.7554** |
| **Order-2** ($\dots + f_{xk} + f_{yk}$)   | **0.6558** | **0.6558** |
| Order-3 full                              | 0.0000 | 0.0000 |

Match to within 0.002 absolute. The bound is exact in the analytic
limit; the 0.002 gap is from 40-bin conditional-mean approximation.

Plot: `lc_anova/results/figures/architectural_floor_analytic.{pdf,png}`.

## 3D Helmholtz gold Sobol (from the first run; HDMR fit re-running)

$d = 4$ on $(x, y, z, k)$. From MC-Saltelli at $N = 300{,}000$:

| Subset | Sobol |
|---|---|
| $S_x$        | 0.031 |
| $S_y$        | 0.029 |
| $S_z$        | 0.027 |
| $S_k$        | **0.208** |
| $S_{xy}$, $S_{xz}$, $S_{yz}$ | 0.020 each (similar) |
| $S_{xk}$, $S_{yk}$, $S_{zk}$ | 0.065 each (similar) |
| $S_{xyz}$    | **0.125** (spatial triplet) |
| $S_{xyk}$, $S_{xzk}$, $S_{yzk}$ | 0.060 each (similar — spatial-pair × k) |
| $\mathbf{S_{xyzk}}$ | **0.137** (full quadruplet) |

Two non-trivial higher-order terms: a spatial triplet ($S_{xyz} = 0.125$)
and a quadruplet ($S_{xyzk} = 0.137$). Method extends to $d=4$ in
principle; HDMR fit currently re-running to confirm.

## What I would do next (when you're back)

1. **Confirm the architectural-complexity finding is what we want.** I
   wrote `architectural_complexity.md` in `draft-cikm/` as the new
   centrepiece section.
2. **Decide what to do with the failed trained-PINN restriction.**
   Options:
   - Drop entirely (analytic projection is the rigorous claim).
   - Pursue more carefully (different optimiser, FiLM-style restricted
     conditioning) as a future-work demonstration.
3. **Re-run synthetic truth at the correct schedule** (5 min, queued).
4. **3D Helm order-3** — wait for current run, then plot.
5. **Decide whether to merge the new abstract** (`abstract_v3.md`) over
   the existing one.

## Files committed

- `8e4a8fb`: analytic-projection headline (script, JSON, plot, code)
- Most recent commit: supporting infrastructure (5 pipelines, 4 plots,
  runner script, plan + status docs, JSON outputs, log)

## Draft updates in `papers/paper-3-lcpinn-hdmr-iclr/draft-cikm/`

- `architectural_complexity.md` (new section, ready)
- `abstract_v3.md` (reframed, ready)
- `FINAL_NUMBERS.md` (consolidated table, mostly filled — synthetic-truth
  + restricted-PINN rows have placeholders pending re-run)
