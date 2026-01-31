"""Configuration for AFQMC single-determinant runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AFQMCConfig:
    dt: float = 0.01
    n_walkers: int = 50
    n_prop_steps: int = 50
    n_blocks: int = 20
    n_eq_steps: int = 50
    ortho_interval: int = 10
    seed: int = 0
    init_noise: float = 0.0
    resample: bool = True
    min_weight: float = 1.0e-3
    max_weight: float = 100.0
    log_interval: int = 1
