"""Loss-conditional PINN framework (equation-agnostic core)."""

from pinns.baseline import FixedWeightPINN, train_fixed_pinn
from pinns.device import device_info, select_device
from pinns.inference import find_best_lambda, find_worst_lambda, sweep_lambda
from pinns.lambda_sampler import LambdaSampler
from pinns.model import LossConditionalPINN
from pinns.training import train_lc_pinn

__all__ = [
    "FixedWeightPINN",
    "LambdaSampler",
    "LossConditionalPINN",
    "device_info",
    "find_best_lambda",
    "find_worst_lambda",
    "select_device",
    "sweep_lambda",
    "train_fixed_pinn",
    "train_lc_pinn",
]
