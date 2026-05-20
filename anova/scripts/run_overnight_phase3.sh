#!/bin/bash
# Phase 3 overnight: fixed compute-cost, 1D Helm compute-cost, additional MC-Sobol,
# and long HDMR stability on 2D Helmholtz.
set -u

cd "$(dirname "$0")/../.."
LOG_DIR="lc_anova/results/overnight_logs"
mkdir -p "$LOG_DIR"

CK_DIR="/Users/anna/Desktop/research/thesis/code/checkpoints"
MASTER="$LOG_DIR/master_phase3.log"

echo "=== phase3 batch started $(date) ===" | tee -a "$MASTER"

# === Compute cost on 2D Helm (3 ReLoBRaLo per-k models) ===
echo "[$(date)] compute cost benchmark 2D Helm (fixed)" | tee -a "$MASTER"
python -m lc_anova.pipelines.compute_cost_benchmark \
    --lc-checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt" \
    --relobralo-pattern 'relobralo_helmholtz_2d_seed0_k.*\.pt' \
    --N 10000 \
    --out lc_anova/results/compute_cost_2d.json \
    > "$LOG_DIR/compute_cost_2d_v2.log" 2>&1

# === Compute cost on 1D Helm (5 ReLoBRaLo per-k models) ===
echo "[$(date)] compute cost benchmark 1D Helm" | tee -a "$MASTER"
python -m lc_anova.pipelines.compute_cost_benchmark \
    --lc-checkpoint "$CK_DIR/lc_pinn_helmholtz_seed0_film_lbfgs.pt" \
    --relobralo-pattern 'relobralo_helmholtz_seed0_k.*\.pt' \
    --N 10000 \
    --out lc_anova/results/compute_cost_1d.json \
    > "$LOG_DIR/compute_cost_1d.log" 2>&1

# === HDMR stability on 2D Helm seed 0 (long) ===
echo "[$(date)] HDMR stability helm2d seed0 (long)" | tee -a "$MASTER"
python -m lc_anova.pipelines.hdmr_stability \
    --pde helm2d \
    --checkpoint "$CK_DIR/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt" \
    --n-runs 5 --phase1-epochs 30 --phase2-epochs 80 --phase3-epochs 200 \
    --tag "helm2d_seed0" \
    > "$LOG_DIR/hdmr_stability_helm2d.log" 2>&1

echo "=== phase3 batch DONE $(date) ===" | tee -a "$MASTER"
