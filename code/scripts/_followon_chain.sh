#!/bin/bash
# Wait for the overnight chain (Burgers + BL FiLM+L-BFGS) to finish, then
# train Schrödinger LC + SA, PI-DeepONet 1D Helm, and a 3D Helm smoke-test.
# All outputs tagged film_lbfgs / smoke so existing JSONs are not overwritten.

set -u
cd "$(dirname "$0")/.."
LOG=logs/followon_chain.log
mkdir -p logs

echo "=== followon chain: started $(date) ===" > "$LOG"

WAIT_PID=83933
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
echo "=== PID $WAIT_PID finished at $(date); starting follow-on ===" >> "$LOG"

# 1. Schrödinger LC FiLM+L-BFGS, 4 seeds, w64.
echo "--- RUN 1: Schrödinger LC FiLM+L-BFGS w64 — start $(date) ---" >> "$LOG"
python -u scripts/lc_pinn_schrodinger.py \
  --seeds 0 1 2 3 --n-epochs 50000 --n-lbfgs 1500 \
  --conditioning film --hidden-width 64 --hidden-depth 4 \
  --tag film_lbfgs >> "$LOG" 2>&1
echo "--- RUN 1 done $(date) (exit $?) ---" >> "$LOG"

# 2. Schrödinger SA-PINN per-α baseline (2 seeds × 3 α).
echo "--- RUN 2: Schrödinger SA-PINN per-α — start $(date) ---" >> "$LOG"
python -u scripts/sa_pinn_schrodinger.py \
  --seeds 0 1 --alpha-trains 0.5 5.0 10.0 \
  --n-adam 5000 --n-lbfgs 2500 >> "$LOG" 2>&1
echo "--- RUN 2 done $(date) (exit $?) ---" >> "$LOG"

# 3. PI-DeepONet 1D Helmholtz, 4 seeds, residual-only.
echo "--- RUN 3: PI-DeepONet 1D Helmholtz — start $(date) ---" >> "$LOG"
python -u scripts/pi_deeponet_helmholtz.py \
  --seeds 0 1 2 3 --n-epochs 50000 --n-lbfgs 1500 \
  --tag matched >> "$LOG" 2>&1
echo "--- RUN 3 done $(date) (exit $?) ---" >> "$LOG"

# 4. 3D Helmholtz smoke-test (1 seed, 2k epochs, no L-BFGS, tiny).
echo "--- RUN 4: 3D Helmholtz smoke-test — start $(date) ---" >> "$LOG"
python -u scripts/lc_pinn_helmholtz_3d.py \
  --seeds 0 --n-epochs 2000 --n-lbfgs 0 \
  --conditioning film --hidden-width 64 --hidden-depth 4 \
  --tag smoke >> "$LOG" 2>&1
echo "--- RUN 4 done $(date) (exit $?) ---" >> "$LOG"

echo "=== followon chain: ALL DONE $(date) ===" >> "$LOG"
