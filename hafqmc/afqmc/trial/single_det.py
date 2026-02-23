"""Single-determinant trial helpers.

This module intentionally stays minimal so the existing high-performance
single-det AFQMC kernels in ``afqmc.py`` remain untouched.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as onp
from jax import numpy as jnp

from ..afqmc_utils import _require_spin_det


def _is_matrix(x: Any) -> bool:
    return isinstance(x, (jnp.ndarray, onp.ndarray)) and getattr(x, "ndim", -1) == 2


def is_single_det_trial(trial: Any) -> bool:
    """Return True if object looks like a spin-separated determinant tuple."""
    if not isinstance(trial, (tuple, list)) or len(trial) != 2:
        return False
    w_up, w_dn = trial
    return _is_matrix(w_up) and _is_matrix(w_dn)


def as_spin_det(trial: Any) -> Tuple[Any, Any]:
    """Validate and return the spin-separated determinant tuple."""
    if not is_single_det_trial(trial):
        raise ValueError("Expected spin-separated single-determinant trial tuple (up, down).")
    _require_spin_det(trial)  # keep existing validation path and message behavior
    return trial


def make_default_trial_state() -> None:
    """Single-det trial has no runtime state."""
    return None
