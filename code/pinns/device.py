"""Pick the best available torch device (CUDA, MPS, or CPU)."""

from __future__ import annotations

import os

import torch


def select_device(prefer: str | None = None) -> torch.device:
    """
    Return a torch.device for training/inference.

    Order when *prefer* is None:
      1. ``PINN_DEVICE`` or ``TORCH_DEVICE`` env var, if set (e.g. ``cuda``, ``mps``, ``cpu``)
      2. NVIDIA GPU: ``cuda`` if ``torch.cuda.is_available()``
      3. Apple Silicon GPU: ``mps`` if supported (PyTorch Metal)
      4. ``cpu``

    On macOS, **CUDA is not available** (NVIDIA drivers); Apple GPU uses **mps**, not cuda.
    """
    env = prefer or os.environ.get("PINN_DEVICE") or os.environ.get("TORCH_DEVICE")
    if env:
        return torch.device(env.strip().lower())

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    """Short human-readable label for logs."""
    if device.type == "cuda" and torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if device.type == "mps":
        return "mps (Apple GPU)"
    return str(device)
