#!/bin/bash
# Phase 2 overnight: per-k Sobol, compute cost, HDMR stability.
set -u

cd "$(dirname "$0")/../.."
LOG_DIR="lc_anova/results/overnight_logs"
mkdir -p "$LOG_DIR"

CK_DIR="/Users/anna/Desktop/research/thesis/code/checkpoints"
MASTER="$LOG_DIR/master_phase2.log"

echo "=== phase2 batch started $(date) ===" | tee -a "$MASTER"

# === Per-k Sobol on 2D Helmholtz, seed 0 (the main checkpoint) ===
for seed in 0 2; do
    echo "[$(date)] per-k Sobol  helm2d seed${seed}" | tee -a "$MASTER"
    python -m lc_anova.pipelines.per_k_sobol \
        --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed${seed}_film_lbfgs_w64.pt" \
        --n-k 11 --N 10000 \
        --tag "per_k_helm2d_seed${seed}" \
        > "$LOG_DIR/per_k_helm2d_seed${seed}.log" 2>&1
done

# === Compute cost benchmark (uses LC-PINN seed 0 + ReLoBRaLo seed 0 per-k models) ===
echo "[$(date)] compute cost benchmark" | tee -a "$MASTER"
python -m lc_anova.pipelines.compute_cost_benchmark \
    --lc-checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt" \
    --relobralo-pattern 'relobralo_helmholtz_2d_seed0_k.*\.pt' \
    --N 10000 \
    > "$LOG_DIR/compute_cost.log" 2>&1

# === HDMR stability (multi-RNG-seed analysis) ===
echo "[$(date)] HDMR stability helm1d" | tee -a "$MASTER"
python -m lc_anova.pipelines.hdmr_stability \
    --pde helm1d \
    --checkpoint "$CK_DIR/lc_pinn_helmholtz_seed0_film_lbfgs.pt" \
    --n-runs 5 --phase1-epochs 30 --phase2-epochs 80 \
    --tag "helm1d_seed0" \
    > "$LOG_DIR/hdmr_stability_helm1d.log" 2>&1

echo "[$(date)] HDMR stability schr1d" | tee -a "$MASTER"
python -m lc_anova.pipelines.hdmr_stability \
    --pde schr1d \
    --checkpoint "$CK_DIR/lc_pinn_schrodinger_seed0_film_lbfgs.pt" \
    --n-runs 5 --phase1-epochs 30 --phase2-epochs 80 \
    --tag "schr1d_seed0" \
    > "$LOG_DIR/hdmr_stability_schr1d.log" 2>&1

echo "=== phase2 batch DONE $(date) ===" | tee -a "$MASTER"
