"""Equation-specific modules for LC-PINN benchmarks.

Each module provides:
  - Problem constants (DIM_PHYS, DIM_LAMBDA, domain parameters)
  - compute_reference_solution() -> dict[float, (x, u)]
  - generate_training_data() -> dict[str, Tensor]
  - compute_losses(model, log_lambda, batch) -> dict[str, Tensor]
  - compute_losses_fixed(model, batch) -> dict[str, Tensor]
  - evaluate(model, log_lambda_or_None, ref_snapshots, device) -> dict[float, float]
  - relative_l2(pred, ref) -> float
"""
