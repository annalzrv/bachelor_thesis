#!/usr/bin/env bash
# Render all plots that depend on overnight outputs. Run after the
# overnight chain completes.

set -u
cd "$(dirname "$0")/../.."

for plot in plot_truncation_floors plot_helm3d_order3 plot_method_ablation plot_architectural_complexity; do
    echo "Rendering $plot"
    python -m lc_anova.plots.$plot || echo "  (skipped — input missing)"
done
