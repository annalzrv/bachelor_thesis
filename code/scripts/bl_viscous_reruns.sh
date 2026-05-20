#!/usr/bin/env bash
# Re-run all four methods on the viscous BL problem (eps=0.01).
# Outputs go to results/{lc_pinn_bl_seeds_viscous,sa_pinn_bl_viscous,relobralo_bl_viscous}.json
# and overwrite checkpoints/ at the same names.

set -e
cd "$(dirname "$0")/.."

EPS="${EPS:-0.01}"
SEEDS="${SEEDS:-0 1 2 3}"

echo "=== LC-PINN BL (viscous, eps=$EPS) ==="
python -u scripts/lc_pinn_bl_seeds.py \
    --seeds $SEEDS \
    --epsilon "$EPS" \
    --tag viscous

echo
echo "=== SA-PINN BL (viscous, eps=$EPS) ==="
python -u scripts/sa_pinn_bl.py \
    --seeds $SEEDS \
    --epsilon "$EPS" \
    --tag viscous

echo
echo "=== ReLoBRaLo BL (viscous, eps=$EPS) ==="
python -u scripts/relobralo_bl.py \
    --seeds $SEEDS \
    --epsilon "$EPS" \
    --tag viscous

echo
echo "Wrote: results/{lc_pinn_bl_seeds,sa_pinn_bl,relobralo_bl}_viscous.json"
