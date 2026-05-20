# bachelor_thesis

Source code, experiments, and LaTeX manuscript for the bachelor's thesis
**"Application of Physics-Informed Neural Networks for Modeling Liquid and
Gas Flow in Porous Media"** at the Faculty of Computer Science, HSE
University.

Author: Anna Alexandrovna Lazareva (group БКНАД222, BSc Computing and Data
Science).
Advisor: Tarakanov Alexander Aleksandrovich, PhD, Associate Professor.

## Repository layout

```
bachelor_thesis/
├── thesis/   LaTeX source of the manuscript (compiles with LuaLaTeX + biber)
├── code/     PINN and LC-PINN training pipeline (PyTorch)
└── anova/    Functional ANOVA / Sobol decomposition module
```

## What is in the thesis

The thesis introduces and studies the **loss-conditional PINN (LC-PINN)**,
a single neural-network architecture trained over a continuous family of
PDE problems indexed by a conditioning vector $\lambda$. Two regimes are
covered uniformly:

- **Parametric-coefficient regime** — $\lambda$ is a physical PDE parameter
  (wavenumber for parametric Helmholtz, harmonic-potential strength for
  Schrödinger). Conditioning uses FiLM scale-and-shift modulation of the
  backbone followed by an L-BFGS refinement pass on a fixed-quadrature
  parameter support.
- **Loss-weight regime** — $\lambda$ is a simplex of loss-component weights
  (PDE / boundary / initial / data). Conditioning is by input
  concatenation; the porous-media application (viscous-regularised
  Buckley–Leverett displacement) is the motivating example.

Two theoretical results frame the empirical study:
- a $\lambda$-invariance result that LC-PINN minimisers recover residual
  minimisers across the full parameter family;
- a local loss-decay advantage that the LC parameterisation enlarges the
  descent set with conditional directions and yields an instantaneous
  loss-decay rate at least as large as the standard PINN at the same
  effective parameter.

A short interpretability section in the experiments chapter applies a
joint $(x, \lambda)$ functional ANOVA decomposition to a trained 2D
Helmholtz LC-PINN. The decomposition produces an a-priori lower bound on
the relative-$L^2$ error of any architecturally simpler solver, verified
to within $0.002$ absolute against the analytic projection at $N = 10^6$.

## Building the manuscript

Requires TeX Live 2024+ with LuaLaTeX and biber.

```
cd thesis
lualatex main && biber main && lualatex main && lualatex main
```

The compiled PDF is `thesis/main.pdf`.

The title page uses PT Serif (system-installed on macOS); on Linux,
install with `apt install fonts-pt-serif` or equivalent.

## Reproducing the experiments

### LC-PINN training (`code/`)

```
cd code
python scripts/lc_pinn_helmholtz.py    --seeds 0 1 2 3 --conditioning film --n-lbfgs 1500 --tag film_lbfgs
python scripts/lc_pinn_schrodinger.py  --seeds 0 1 2 3 --conditioning film --n-lbfgs 1500 --tag film_lbfgs
python scripts/lc_pinn_helmholtz_2d.py --seeds 0 1 2 3 --conditioning film --hidden-width 64 --n-lbfgs 1500 --tag film_lbfgs_w64
python scripts/lc_pinn_burgers_seeds.py
python scripts/lc_pinn_bl_seeds.py
```

Checkpoints are written to `code/checkpoints/` (gitignored, regenerate
locally). Per-seed results are written to `code/results/*.json`.

### Functional ANOVA decomposition (`anova/`)

```
cd anova
python -m lc_anova.pipelines.helmholtz_2d \
    --checkpoint ../code/checkpoints/lc_pinn_helmholtz_2d_seed0_film_lbfgs_w64.pt \
    --out-dir lc_anova/results
python -m lc_anova.pipelines.mc_sobol_analytic
python -m lc_anova.pipelines.architectural_floor_analytic
```

The ANOVA pipelines consume the trained LC-PINN checkpoints from
`code/checkpoints/` and produce the Sobol-index JSONs and decomposition
plots under `anova/results/`.

## Hardware

All experiments were run on an Apple M4 Max GPU via the PyTorch MPS
backend. The code also runs on CUDA without modification.

## License

Code released for academic use. See the thesis manuscript for citations
of all external methods and datasets used.
