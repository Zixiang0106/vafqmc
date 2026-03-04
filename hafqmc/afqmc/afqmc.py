"""Public AFQMC API and trial dispatch."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any, Optional

from ml_collections import ConfigDict

from ..hamiltonian import Hamiltonian
from .afqmc_config import (
    AFQMCConfig,
    cassci_trial_default,
    default as afqmc_default,
    stochastic_trial_default,
)
from .afqmc_utils import build_hamiltonian_pickle, load_hamiltonian
from .driver.custom import run_afqmc_custom
from .driver.det import run_afqmc_det
from .trial.cassci import CASSCITrial
from .trial.single_det import as_spin_det, is_single_det_trial
from .trial.stochastic import VAFQMCTrial


def _get_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    if logger is not None:
        return logger
    logger = logging.getLogger("hafqmc.afqmc")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _to_cfg(cfg: Any | None) -> ConfigDict:
    if cfg is None:
        return AFQMCConfig.default()
    if isinstance(cfg, ConfigDict):
        return cfg.copy_and_resolve_references()
    if isinstance(cfg, Mapping):
        merged = afqmc_default()
        for k, v in cfg.items():
            merged[k] = v
        return merged
    return cfg


def _runtime_cfg(cfg: ConfigDict) -> ConfigDict:
    """Attach flat aliases expected by existing driver kernels."""
    out = cfg.copy_and_resolve_references()

    # legacy flat top-level support (if user passes AFQMCConfig(dt=...))
    p = out.get("propagation", None)
    if p is None:
        p = out.propagation = afqmc_default().propagation
    lg = out.get("log", None)
    if lg is None:
        lg = out.log = afqmc_default().log
    pop = out.get("pop_control", None)
    if pop is None:
        pop = out.pop_control = afqmc_default().pop_control

    if "dt" in out:
        p.dt = out.dt
    if "n_walkers" in out:
        p.n_walkers = out.n_walkers
    if "n_prop_steps" in out:
        p.n_prop_steps = out.n_prop_steps
    if "n_ene_blocks" in out:
        p.n_ene_blocks = out.n_ene_blocks
    if "n_sr_blocks" in out:
        p.n_sr_blocks = out.n_sr_blocks
    if "n_blocks" in out:
        p.n_blocks = out.n_blocks
    if "n_eq_steps" in out:
        p.n_eq_steps = out.n_eq_steps
    if "ortho_interval" in out:
        p.ortho_interval = out.ortho_interval
    if "log_interval" in out:
        p.log_interval = out.log_interval
        lg.block_interval = out.log_interval

    if "init_noise" in out:
        pop.init_noise = out.init_noise
    if "resample" in out:
        pop.resample = out.resample
    if "pop_control_freq" in out:
        pop.freq = out.pop_control_freq
    if "pop_control_log_stats" in out:
        pop.log_stats = out.pop_control_log_stats
        lg.pop_control_stats = out.pop_control_log_stats
    if "min_weight" in out:
        pop.min_weight = out.min_weight
    if "max_weight" in out:
        pop.max_weight = out.max_weight

    # logging aliases
    if "log_enabled" in out:
        lg.enabled = out.log_enabled
    if "log_block_interval" in out:
        lg.block_interval = out.log_block_interval
    if "log_equil_interval" in out:
        lg.equil_interval = out.log_equil_interval
    if "log_equil_n_print" in out:
        lg.equil_n_print = out.log_equil_n_print

    st = out.get("stochastic_trial", None)
    if st is None and str(out.get("trial_type", "single_det")).lower() in (
        "stochastic",
        "vafqmc",
        "vtrial",
        "custom",
    ):
        st = out.stochastic_trial = stochastic_trial_default()
    ct = out.get("cassci_trial", None)
    if ct is None and str(out.get("trial_type", "single_det")).lower() in (
        "cassci",
        "casci",
        "cas",
        "multi_det",
        "multidet",
    ):
        ct = out.cassci_trial = cassci_trial_default()

    out.dt = float(p.dt)
    out.n_walkers = int(p.n_walkers)
    out.n_prop_steps = int(p.n_prop_steps)
    out.n_ene_blocks = int(p.get("n_ene_blocks", 1))
    out.n_sr_blocks = int(p.get("n_sr_blocks", 1))
    out.n_blocks = int(p.n_blocks)
    out.n_eq_steps = int(p.n_eq_steps)
    out.ortho_interval = int(p.ortho_interval)

    out.log_enabled = bool(lg.get("enabled", True))
    out.log_block_interval = int(lg.get("block_interval", p.get("log_interval", 1)))
    out.log_equil_interval = int(lg.get("equil_interval", 0))
    out.log_equil_n_print = int(lg.get("equil_n_print", 5))
    # keep legacy flat key for backward compatibility in external scripts
    out.log_interval = int(out.log_block_interval)

    out.init_noise = float(pop.init_noise)
    out.resample = bool(pop.resample)
    out.pop_control_freq = int(pop.get("freq", 0))
    out.pop_control_log_stats = bool(lg.get("pop_control_stats", pop.get("log_stats", False)))
    out.min_weight = float(pop.min_weight)
    out.max_weight = float(pop.max_weight)

    vis_cfg = out.get("visualization", False)
    is_vis_dict_like = (
        hasattr(vis_cfg, "get")
        and not isinstance(vis_cfg, (bool, int, float, str))
    )
    if is_vis_dict_like:
        out.visualization = bool(vis_cfg.get("enabled", False))
        out.visualization_refresh_every = int(vis_cfg.get("refresh_every", 1))
        out.visualization_show = bool(vis_cfg.get("show", True))
        out.visualization_save_path = vis_cfg.get("save_path", None)
    else:
        out.visualization = bool(vis_cfg)
        out.visualization_refresh_every = 1
        out.visualization_show = True
        out.visualization_save_path = None

    if st is None:
        out.n_measure_samples = 1
    else:
        if "n_measure_samples" in out:
            st.n_measure_samples = out.n_measure_samples
        out.n_measure_samples = int(st.get("n_measure_samples", 1))

    if "trial_kind" in out:
        out.trial_type = out.trial_kind

    return out


def _stochastic_sampler_kwargs(st_cfg: Any) -> dict[str, Any]:
    sampler = st_cfg.sampler
    kwargs = dict(getattr(sampler, "kwargs", {}) or {})
    if str(sampler.name).lower() in ("hmc", "hamiltonian", "hybrid"):
        kwargs.setdefault("dt", float(sampler.dt))
        kwargs.setdefault("length", float(sampler.length))
    return kwargs


def _build_trial_from_config(hamil: Hamiltonian, cfg: ConfigDict) -> Any:
    kind = str(cfg.trial_type).lower()
    if kind in ("single_det", "single-det", "det", "sd", "reference", "hf"):
        return hamil.wfn0

    if kind in ("stochastic", "vafqmc", "vtrial", "custom"):
        st = cfg.get("stochastic_trial", None)
        if st is None:
            st = stochastic_trial_default()
            cfg.stochastic_trial = st

        if not st.checkpoint:
            raise ValueError(
                "cfg.stochastic_trial.checkpoint is required when trial_type is stochastic/vafqmc."
            )

        trial_seed = int(cfg.seed if st.trial_seed is None else st.trial_seed)
        return VAFQMCTrial.from_hparams_checkpoint(
            hamil,
            st.checkpoint,
            hparams_path=st.hparams_path,
            n_samples=int(st.n_samples),
            burn_in=int(st.burn_in),
            sampler_name=str(st.sampler.name),
            sampler_kwargs=_stochastic_sampler_kwargs(st),
            sampling_target=str(st.sampling_target),
            logdens_floor=float(st.logdens_floor),
            sample_update_steps=int(st.sample_update_steps),
            init_walkers_from_trial=bool(st.init_walkers_from_trial),
            init_walkers_burn_in=int(st.init_walkers_burn_in),
            max_prop=st.max_prop,
            seed=trial_seed,
        )

    if kind in ("cassci", "casci", "cas", "multi_det", "multidet"):
        ct = cfg.get("cassci_trial", None)
        if ct is None:
            ct = cassci_trial_default()
            cfg.cassci_trial = ct
        if not ct.path:
            raise ValueError(
                "cfg.cassci_trial.path is required when trial_type is cassci."
            )
        return CASSCITrial.from_file(
            str(ct.path),
            n_det=int(ct.n_det) if ct.get("n_det", None) is not None else None,
            coeff_cutoff=float(ct.get("coeff_cutoff", 0.0)),
            normalize_coeffs=bool(ct.get("normalize_coeffs", True)),
            init_mode=str(ct.get("init_mode", "sample_coeff")),
        )

    raise ValueError(f"Unknown trial_type: {cfg.trial_type}")


def afqmc_energy(
    hamil: Hamiltonian,
    trial: Optional[Any] = None,
    *,
    cfg: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    return_state: bool = False,
    return_blocks: bool = False,
):
    """Run AFQMC energy estimation.

    Trial dispatch:
    - if ``trial`` is provided, use it directly
    - else use ``cfg.trial_type`` + ``cfg.stochastic_trial``
    """
    cfg_user = _to_cfg(cfg)
    cfg_runtime = _runtime_cfg(cfg_user)
    logger = _get_logger(logger)
    start = time.time()

    if trial is None:
        trial = _build_trial_from_config(hamil, cfg_runtime)

    if is_single_det_trial(trial):
        trial = as_spin_det(trial)
        return run_afqmc_det(
            hamil,
            trial,
            cfg_runtime,
            logger,
            start,
            return_state=return_state,
            return_blocks=return_blocks,
        )

    return run_afqmc_custom(
        hamil,
        trial,
        cfg_runtime,
        logger,
        start,
        return_state=return_state,
        return_blocks=return_blocks,
    )


def afqmc_energy_from_pickle(
    hamil_path: str,
    *,
    cfg: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    trial: Optional[Any] = None,
    return_state: bool = False,
    return_blocks: bool = False,
):
    hamil = load_hamiltonian(hamil_path)
    return afqmc_energy(
        hamil,
        trial=trial,
        cfg=cfg,
        logger=logger,
        return_state=return_state,
        return_blocks=return_blocks,
    )


def afqmc_energy_from_checkpoint(
    hamil_path: str,
    checkpoint_path: str,
    *,
    hparams_path: str = "hparams.yml",
    cfg: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    n_walkers: Optional[int] = None,
    n_samples: Optional[int] = None,
    burn_in: Optional[int] = None,
    sampler_name: Optional[str] = None,
    sampler_kwargs: Optional[dict] = None,
    sampling_target: Optional[str] = None,
    logdens_floor: Optional[float] = None,
    sample_update_steps: Optional[int] = None,
    init_walkers_from_trial: Optional[bool] = None,
    init_walkers_burn_in: Optional[int] = None,
    max_prop: Optional[Any] = None,
    seed: Optional[int] = None,
    return_state: bool = False,
    return_blocks: bool = False,
):
    """Compatibility wrapper: map args into ConfigDict-style AFQMC config."""
    cfg = _to_cfg(cfg)
    cfg.trial_type = "stochastic"
    if cfg.get("stochastic_trial", None) is None:
        cfg.stochastic_trial = stochastic_trial_default()

    cfg.stochastic_trial.checkpoint = checkpoint_path
    cfg.stochastic_trial.hparams_path = hparams_path

    if n_walkers is not None:
        cfg.propagation.n_walkers = int(n_walkers)
    if n_samples is not None:
        cfg.stochastic_trial.n_samples = int(n_samples)
    if burn_in is not None:
        cfg.stochastic_trial.burn_in = int(burn_in)
    if sampler_name is not None:
        cfg.stochastic_trial.sampler.name = str(sampler_name)
    if sampler_kwargs is not None:
        cfg.stochastic_trial.sampler.kwargs = dict(sampler_kwargs)
        if "dt" in sampler_kwargs:
            cfg.stochastic_trial.sampler.dt = float(sampler_kwargs["dt"])
        if "length" in sampler_kwargs:
            cfg.stochastic_trial.sampler.length = float(sampler_kwargs["length"])
    if sampling_target is not None:
        cfg.stochastic_trial.sampling_target = str(sampling_target)
    if logdens_floor is not None:
        cfg.stochastic_trial.logdens_floor = float(logdens_floor)
    if sample_update_steps is not None:
        cfg.stochastic_trial.sample_update_steps = int(sample_update_steps)
    if init_walkers_from_trial is not None:
        cfg.stochastic_trial.init_walkers_from_trial = bool(init_walkers_from_trial)
    if init_walkers_burn_in is not None:
        cfg.stochastic_trial.init_walkers_burn_in = int(init_walkers_burn_in)
    if max_prop is not None:
        cfg.stochastic_trial.max_prop = max_prop
    if seed is not None:
        cfg.seed = int(seed)
        cfg.stochastic_trial.trial_seed = int(seed)

    return afqmc_energy_from_pickle(
        hamil_path,
        cfg=cfg,
        logger=logger,
        return_state=return_state,
        return_blocks=return_blocks,
    )


__all__ = [
    "AFQMCConfig",
    "afqmc_energy",
    "afqmc_energy_from_checkpoint",
    "afqmc_energy_from_pickle",
    "build_hamiltonian_pickle",
]
