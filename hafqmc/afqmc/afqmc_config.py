"""AFQMC ConfigDict presets.

This module follows the same style as ``hafqmc.config``:
- ``default() -> ConfigDict``
- grouped keys for readability
- example constructors for common trial types
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ml_collections import ConfigDict


def _cfg(data: dict[str, Any]) -> ConfigDict:
    return ConfigDict(data, type_safe=False, convert_dict=True)


def propagation_default() -> ConfigDict:
    return _cfg(
        {
            "dt": 0.01,
            "n_walkers": 100,
            "n_block_steps": 50,
            "n_ene_measurements": 1,
            "n_blocks": 20,
            "n_eq_steps": 150,
            "ortho_freq": 10,
        }
    )


def log_default() -> ConfigDict:
    return _cfg(
        {
            "enabled": True,
            "block_freq": 1,
            "equil_freq": 0,
            "pop_control_stats": False,
        }
    )


def pop_control_default() -> ConfigDict:
    return _cfg(
        {
            "init_noise": 0.0,
            "resample": True,
            "freq": 10,
            "min_weight": 1.0e-3,
            "max_weight": 100.0,
        }
    )


def output_default() -> ConfigDict:
    return _cfg(
        {
            "write_raw": False,
            "raw_path": "raw.dat",
            "write_hparams": False,
            "hparams_path": "afqmc_hparams.yml",
            "visualization": {
                "enabled": False,
                "refresh_every": 1,
                "show": False,
                "save_path": "eblock_afqmc.png",
            },
        }
    )


def stochastic_trial_default() -> ConfigDict:
    return _cfg(
        {
            "checkpoint": None,
            "hparams_path": "hparams.yml",
            "n_samples": 20,
            "burn_in": 1000,
            "sampler": {
                "name": "hmc",
                "dt": 0.1,
                "length": 1.0,
                "kwargs": {},
            },
            "sampling_target": "walker_overlap",
            "logdens_floor": -60.0,
            "sample_update_steps": 1,
            "init_walkers_from_trial": False,
            "init_walkers_burn_in": 0,
            "init_walkers_chains_per_walker": 0,
            "init_walkers_infer_steps": 10,
            "max_prop": None,
            "trial_seed": None,
            "n_measure_samples": 20,
        }
    )


def cassci_trial_default() -> ConfigDict:
    return _cfg(
        {
            "path": None,
            "n_det": 20,
            "coeff_cutoff": 0.0,
            "normalize_coeffs": True,
            "init_mode": "sample_coeff",
        }
    )


def default() -> ConfigDict:
    return _cfg(
        {
            "propagation": propagation_default(),
            "log": log_default(),
            "pop_control": pop_control_default(),
            "output": output_default(),
            "trial_type": "single_det",
            "stochastic_trial": None,
            "cassci_trial": None,
            "seed": 0,
        }
    )


def single_det_example() -> ConfigDict:
    cfg = default()
    cfg.trial_type = "single_det"
    cfg.stochastic_trial = None
    return cfg


def stochastic_example() -> ConfigDict:
    cfg = default()
    cfg.trial_type = "stochastic"
    cfg.stochastic_trial = stochastic_trial_default()
    return cfg


def cassci_example() -> ConfigDict:
    cfg = default()
    cfg.trial_type = "cassci"
    cfg.cassci_trial = cassci_trial_default()
    cfg.stochastic_trial = None
    return cfg


def _deep_update(dst: ConfigDict, src: Mapping[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, Mapping):
            cur = dst.get(key, None)
            if cur is None or not isinstance(cur, ConfigDict):
                dst[key] = _cfg({})
            _deep_update(dst[key], value)
        else:
            dst[key] = value


class AFQMCConfig:
    """Factory-style wrapper for ConfigDict AFQMC configs.

    Usage:
        cfg = AFQMCConfig()
        cfg = AFQMCConfig(dt=0.01)  # legacy flat key still accepted in runner
        cfg = AFQMCConfig.stochastic_example()
    """

    def __new__(cls, **kwargs: Any) -> ConfigDict:
        cfg = default()
        if kwargs:
            _deep_update(cfg, kwargs)
        return cfg

    @staticmethod
    def default() -> ConfigDict:
        return default()

    @staticmethod
    def single_det_example() -> ConfigDict:
        return single_det_example()

    @staticmethod
    def stochastic_example() -> ConfigDict:
        return stochastic_example()

    @staticmethod
    def cassci_example() -> ConfigDict:
        return cassci_example()


__all__ = [
    "AFQMCConfig",
    "cassci_example",
    "cassci_trial_default",
    "default",
    "log_default",
    "output_default",
    "pop_control_default",
    "propagation_default",
    "single_det_example",
    "stochastic_example",
    "stochastic_trial_default",
]
