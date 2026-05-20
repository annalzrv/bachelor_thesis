#!/bin/bash
# Overnight batch: HDMR + MC-Sobol on three parametric PDEs, multiple seeds.
# Runs sequentially because MPS doesn't share well between processes.
set -u

cd "$(dirname "$0")/../.."
LOG_DIR="lc_anova/results/overnight_logs"
mkdir -p "$LOG_DIR"

CK_DIR="/Users/anna/Desktop/research/thesis/code/checkpoints"

echo "=== overnight batch started $(date) ===" | tee "$LOG_DIR/master.log"

run_d2_pipeline() {
    local pde="$1"
    local checkpoint="$2"
    local tag="$3"
    echo "[$(date)] HDMR  $pde  $tag" | tee -a "$LOG_DIR/master.log"
    python -m lc_anova.pipelines.pde1d \
        --pde "$pde" --checkpoint "$checkpoint" \
        --fourier --hidden 128 --layers 3 --num-freqs 6 \
        --phase1-epochs 40 --phase2-epochs 120 \
        --tag "$tag" \
        > "$LOG_DIR/hdmr_${tag}.log" 2>&1
}

run_d3_pipeline() {
    local checkpoint="$1"
    local tag="$2"
    echo "[$(date)] HDMR  helm2d  $tag" | tee -a "$LOG_DIR/master.log"
    python -m lc_anova.pipelines.helmholtz_2d \
        --checkpoint "$checkpoint" \
        --max-order 3 --fourier --hidden 128 --layers 3 --num-freqs 6 \
        --phase1-epochs 40 --phase2-epochs 80 --phase3-epochs 400 \
        --tag "$tag" \
        > "$LOG_DIR/hdmr_${tag}.log" 2>&1
}

run_mc_sobol_2d() {
    local checkpoint="$1"
    local tag="$2"
    echo "[$(date)] MC-Sobol  helm2d  $tag" | tee -a "$LOG_DIR/master.log"
    python -m lc_anova.pipelines.mc_sobol_helmholtz_2d \
        --checkpoint "$checkpoint" --N 30000 --tag "$tag" \
        > "$LOG_DIR/mc_${tag}.log" 2>&1
}

# === 1D Helmholtz: 4 LC-PINN seeds × Fourier HDMR ===
for seed in 0 1 2 3; do
    run_d2_pipeline helmholtz \
        "$CK_DIR/lc_pinn_helmholtz_seed${seed}_film_lbfgs.pt" \
        "helm1d_seed${seed}"
done

# === Schrödinger: 4 LC-PINN seeds × Fourier HDMR ===
for seed in 0 1 2 3; do
    run_d2_pipeline schrodinger \
        "$CK_DIR/lc_pinn_schrodinger_seed${seed}_film_lbfgs.pt" \
        "schr1d_seed${seed}"
done

# === 2D Helmholtz: seeds 2, 3 (seed 0 already done; seed 1 NaN'd in training) ===
for seed in 2 3; do
    run_d3_pipeline \
        "$CK_DIR/lc_pinn_helmholtz_2d_seed${seed}_film_lbfgs_w64.pt" \
        "helm2d_fourier_seed${seed}"
done

# === MC-Sobol cross-checks on 2D Helmholtz seeds 1, 3 (already have 0, 2) ===
for seed in 1 3; do
    run_mc_sobol_2d \
        "$CK_DIR/lc_pinn_helmholtz_2d_seed${seed}_film_lbfgs_w64.pt" \
        "mc_sobol_seed${seed}"
done

echo "=== overnight batch DONE $(date) ===" | tee -a "$LOG_DIR/master.log"
