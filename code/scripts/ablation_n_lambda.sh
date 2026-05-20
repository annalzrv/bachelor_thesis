#!/usr/bin/env bash
# n_lambda_samples ablation: retrain LC-PINN Burgers at n ∈ {1, 4, 16} with
# a shorter budget (25k epochs) for clean head-to-head comparison.
# Outputs go to results/lc_pinn_burgers_ablation_n{N}.json.

set -e
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1}"
N_EPOCHS="${N_EPOCHS:-25000}"

for N_LAM in 1 4 16; do
    echo "=== n_lambda_samples = $N_LAM ==="
    python -u scripts/lc_pinn_burgers_seeds.py \
        --seeds $SEEDS \
        --n-epochs "$N_EPOCHS" \
        --n-lambda-samples "$N_LAM" \
        --tag "ablation_n${N_LAM}"
    echo
done

echo "Wrote: results/lc_pinn_burgers_ablation_n{1,4,16}.json"
