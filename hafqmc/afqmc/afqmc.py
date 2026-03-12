"""Public AFQMC API and trial dispatch."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Optional

from ml_collections import ConfigDict

from ..hamiltonian import Hamiltonian
from ..utils import cfg_to_yaml
from .afqmc_config import (
    AFQMCConfig,
    cassci_trial_default,
    default as afqmc_default,
    stochastic_trial_default,
)
from .utils import build_hamiltonian_pickle, load_hamiltonian
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
    otp = out.get("output", None)
    if otp is None:
        otp = out.output = afqmc_default().output
    vis = otp.get("visualization", None)
    if vis is None:
        vis = otp.visualization = afqmc_default().output.visualization

    if "dt" in out:
        p.dt = out.dt
    if "n_walkers" in out:
        p.n_walkers = out.n_walkers
    if "n_block_steps" in out:
        p.n_block_steps = out.n_block_steps
    # Backward-compatible aliases (old names).
    if "n_prop_steps" in out:
        p.n_block_steps = out.n_prop_steps
    if "n_ene_measurements" in out:
        p.n_ene_measurements = out.n_ene_measurements
    # Backward-compatible aliases (old names).
    if "n_ene_blocks" in out:
        p.n_ene_measurements = out.n_ene_blocks
    if "n_blocks" in out:
        p.n_blocks = out.n_blocks
    if "n_eq_steps" in out:
        p.n_eq_steps = out.n_eq_steps
    if "ortho_freq" in out:
        p.ortho_freq = out.ortho_freq

    if "init_noise" in out:
        pop.init_noise = out.init_noise
    if "resample" in out:
        pop.resample = out.resample
    if "resample_normalize_wsum" in out:
        pop.normalize_wsum_after_resample = out.resample_normalize_wsum
    if "pop_control_freq" in out:
        pop.freq = out.pop_control_freq
    if "pop_control_log_stats" in out:
        lg.pop_control_stats = out.pop_control_log_stats
    if "min_weight" in out:
        pop.min_weight = out.min_weight
    if "max_weight" in out:
        pop.max_weight = out.max_weight
    if "write_raw" in out:
        otp.write_raw = out.write_raw
    if "raw_path" in out:
        otp.raw_path = out.raw_path
    if "write_hparams" in out:
        otp.write_hparams = out.write_hparams
    if "hparams_path" in out:
        otp.hparams_path = out.hparams_path
    # Output visualization aliases
    if "output_visualization_enabled" in out:
        vis.enabled = out.output_visualization_enabled
    if "output_visualization_refresh_every" in out:
        vis.refresh_every = out.output_visualization_refresh_every
    if "output_visualization_show" in out:
        vis.show = out.output_visualization_show
    if "output_visualization_save_path" in out:
        vis.save_path = out.output_visualization_save_path

    # logging aliases
    if "log_enabled" in out:
        lg.enabled = out.log_enabled
    if "log_block_freq" in out:
        lg.block_freq = out.log_block_freq
    if "log_equil_freq" in out:
        lg.equil_freq = out.log_equil_freq
    # Backward-compatible top-level visualization aliases.
    if "visualization" in out:
        vis.enabled = out.visualization
    if "visualization_refresh_every" in out:
        vis.refresh_every = out.visualization_refresh_every
    if "visualization_show" in out:
        vis.show = out.visualization_show
    if "visualization_save_path" in out:
        vis.save_path = out.visualization_save_path

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
    out.n_block_steps = int(p.get("n_block_steps", p.get("n_prop_steps", 50)))
    out.n_ene_measurements = int(
        p.get("n_ene_measurements", p.get("n_ene_blocks", 1))
    )
    # Backward-compatible flat aliases for internals/external scripts.
    out.n_prop_steps = int(out.n_block_steps)
    out.n_ene_blocks = int(out.n_ene_measurements)
    out.n_blocks = int(p.n_blocks)
    out.n_eq_steps = int(p.n_eq_steps)
    out.ortho_freq = int(p.ortho_freq)

    out.log_enabled = bool(lg.get("enabled", True))
    out.log_block_freq = int(lg.get("block_freq", 1))
    out.log_equil_freq = int(lg.get("equil_freq", 0))

    out.init_noise = float(pop.init_noise)
    out.resample = bool(pop.resample)
    out.resample_normalize_wsum = bool(pop.get("normalize_wsum_after_resample", False))
    out.pop_control_freq = int(pop.get("freq", 0))
    out.pop_control_log_stats = bool(lg.get("pop_control_stats", False))
    out.min_weight = float(pop.min_weight)
    out.max_weight = float(pop.max_weight)
    out.output_write_raw = bool(otp.get("write_raw", False))
    out.output_raw_path = str(otp.get("raw_path", "raw.dat"))
    out.output_write_hparams = bool(otp.get("write_hparams", False))
    out.output_hparams_path = str(otp.get("hparams_path", "afqmc_hparams.yml"))
    out.output_visualization_enabled = bool(vis.get("enabled", False))
    out.output_visualization_refresh_every = int(vis.get("refresh_every", 1))
    out.output_visualization_show = bool(vis.get("show", False))
    out.output_visualization_save_path = str(
        vis.get("save_path", "eblock_afqmc.png")
    )

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


def _save_afqmc_hparams(cfg: ConfigDict, path: str) -> None:
    p = Path(path)
    if p.parent and str(p.parent) != "":
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(cfg_to_yaml(cfg))


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
            local_energy_chunk_size=int(st.get("local_energy_chunk_size", 0)),
            init_walkers_from_trial=bool(st.init_walkers_from_trial),
            init_walkers_burn_in=int(st.init_walkers_burn_in),
            init_walkers_chains_per_walker=int(st.get("init_walkers_chains_per_walker", 0)),
            init_walkers_infer_steps=int(st.get("init_walkers_infer_steps", 10)),
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

    if bool(getattr(cfg_runtime, "output_write_hparams", False)):
        _save_afqmc_hparams(cfg_user, str(cfg_runtime.output_hparams_path))

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
