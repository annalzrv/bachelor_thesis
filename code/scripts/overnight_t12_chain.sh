#!/bin/bash
# Overnight chain: Tier 1 (PI-DeepONet seed 4 on 1D Helm + 4 seeds on Schrödinger)
# then Tier 2 (3D Helmholtz LC FiLM+L-BFGS + per-k SA + per-k ReLoBRaLo).
# Each step writes a fresh JSON in results/ — never overwrites prior data.
set -u

cd "$(dirname "$0")/.."
LOG="logs/overnight_t12_chain.log"
mkdir -p logs

run() {
    local name="$1"; shift
    echo "" >> "$LOG"
    echo "=== START $name $(date) ===" >> "$LOG"
    "$@" >> "$LOG" 2>&1
    local ec=$?
    echo "=== END $name $(date) (exit $ec) ===" >> "$LOG"
    return $ec
}

echo "=== chain start $(date) ===" >> "$LOG"

# --- Tier 1.1: PI-DeepONet 1D Helmholtz, additional seed 4 (matched_v2 tag) ---
run "T1.1 PI-DeepONet 1D Helm seed 4" \
    python -u scripts/pi_deeponet_helmholtz.py \
        --seeds 4 --n-epochs 50000 --n-lbfgs 1500 --tag matched_v2

# --- Tier 1.2: PI-DeepONet 1D Schrödinger, 4 seeds, matched config ---
run "T1.2 PI-DeepONet Schrödinger 4 seeds" \
    python -u scripts/pi_deeponet_schrodinger.py \
        --seeds 0 1 2 3 --n-epochs 50000 --n-lbfgs 1500 --tag matched

# --- Tier 2.1: 3D Helmholtz LC-PINN, FiLM + L-BFGS, 4 seeds ---
run "T2.1 LC-PINN 3D Helm FiLM+LBFGS 4 seeds" \
    python -u scripts/lc_pinn_helmholtz_3d.py \
        --seeds 0 1 2 3 --n-epochs 25000 --conditioning film \
        --n-lbfgs 1500 --hidden-width 64 --tag film_lbfgs

# --- Tier 2.2: 3D Helmholtz SA-PINN per-k, 3 k-trains, 2 seeds ---
run "T2.2 SA-PINN 3D Helm per-k 3k×2s" \
    python -u scripts/sa_pinn_helmholtz_3d.py \
        --seeds 0 1 --k-trains 1.0 3.0 5.0 --n-adam 5000 --n-lbfgs 2500

# --- Tier 2.3: 3D Helmholtz ReLoBRaLo per-k, 3 k-trains, 2 seeds ---
run "T2.3 ReLoBRaLo 3D Helm per-k 3k×2s" \
    python -u scripts/relobralo_helmholtz_3d.py \
        --seeds 0 1 --k-trains 1.0 3.0 5.0 --n-epochs 25000

echo "" >> "$LOG"
echo "=== chain ALL DONE $(date) ===" >> "$LOG"
