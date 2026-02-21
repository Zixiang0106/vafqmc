"""AFQMC driver (single-determinant, phaseless)."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import jax
from jax import lax, numpy as jnp

from ..hamiltonian import Hamiltonian
from ..propagator import orthonormalize
from .afqmc_config import AFQMCConfig
from .afqmc_utils import (
    PropagationData,
    apply_trotter,
    build_hamiltonian_pickle,
    build_propagation_data,
    calc_force_bias,
    calc_rdm_batch,
    calc_local_energy_batch,
    calc_slov_batch,
    load_hamiltonian,
    _is_custom_trial,
    _spin_sum_rdm,
)
from .vafqmc_trial import VAFQMCTrial
from .walker import AFQMCState, init_walkers, maybe_orthonormalize, stochastic_reconfiguration


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


def _propagate_step(
    hamil: Hamiltonian,
    trial: Any,
    state: AFQMCState,
    prop_data: PropagationData,
    cfg: AFQMCConfig,
) -> AFQMCState:
    n_walkers = state.weights.shape[0]
    nchol = prop_data.vhs.shape[0]

    key, subkey = jax.random.split(state.key)
    fields = jax.random.normal(
        subkey, (n_walkers, nchol), dtype=jnp.real(prop_data.vhs).dtype
    )

    force_bias = calc_force_bias(hamil, trial, state.walkers, prop_data)
    field_shifts = -prop_data.sqrt_dt * (1.0j * force_bias - prop_data.mf_shifts)
    shifted_fields = fields - field_shifts

    shift_term = jnp.einsum("wk,k->w", shifted_fields, prop_data.mf_shifts)
    fb_term = jnp.einsum("wk,wk->w", fields, field_shifts) - 0.5 * jnp.einsum(
        "wk,wk->w", field_shifts, field_shifts
    )

    walkers_new = apply_trotter(state.walkers, shifted_fields, prop_data)
    sign_prop, logov_prop = calc_slov_batch(trial, walkers_new)

    valid = jnp.isfinite(state.logov)
    log_ratio = jnp.where(valid, logov_prop - state.logov, -jnp.inf)
    sign_ratio = jnp.where(valid, sign_prop / state.sign, 0.0 + 0.0j)
    ovlp_ratio = sign_ratio * jnp.exp(log_ratio)

    imp = jnp.exp(
        -prop_data.sqrt_dt * shift_term
        + fb_term
        + prop_data.dt * (state.pop_control_shift + prop_data.h0_prop)
    ) * ovlp_ratio
    theta = jnp.angle(jnp.exp(-prop_data.sqrt_dt * shift_term) * ovlp_ratio)
    imp_ph = jnp.abs(imp) * jnp.cos(theta)
    imp_ph = jnp.where(jnp.isnan(imp_ph), 0.0, imp_ph)
    imp_ph = jnp.where(imp_ph < 0.0, 0.0, imp_ph)
    imp_ph = jnp.where(imp_ph < cfg.min_weight, 0.0, imp_ph)
    imp_ph = jnp.where(imp_ph > cfg.max_weight, 0.0, imp_ph)

    # For tethered custom trials, update per-walker sample pools immediately
    # after walker propagation, then apply an extra hand-off phaseless factor.
    sign_new, logov_new = sign_prop, logov_prop
    walker_fields = state.walker_fields
    if _is_custom_trial(trial) and hasattr(trial, "update_tethered_samples"):
        handoff, sign_new, logov_new = trial.update_tethered_samples(walkers_new)
        if hasattr(trial, "walker_fields") and getattr(trial, "walker_fields") is not None:
            walker_fields = trial.walker_fields
        handoff = jnp.real(handoff)
        handoff = jnp.where(jnp.isnan(handoff), 0.0, handoff)
        handoff = jnp.where(handoff < 0.0, 0.0, handoff)
        handoff = jnp.where(handoff < cfg.min_weight, 0.0, handoff)
        handoff = jnp.where(handoff > cfg.max_weight, 0.0, handoff)
        imp_ph = imp_ph * handoff

    weights = state.weights * imp_ph
    weights = jnp.where(weights > cfg.max_weight, 0.0, weights)
    wsum = jnp.maximum(jnp.sum(weights), 1.0e-12)
    pop_control_shift = state.e_estimate - 0.1 * jnp.log(wsum / n_walkers) / prop_data.dt

    return AFQMCState(
        walkers=walkers_new,
        weights=weights,
        sign=sign_new,
        logov=logov_new,
        key=key,
        e_estimate=state.e_estimate,
        pop_control_shift=pop_control_shift,
        walker_fields=walker_fields,
    )


def _calc_mixed_energy(hamil: Hamiltonian, trial: Any, state: AFQMCState) -> Array:
    e_loc = jnp.real(calc_local_energy_batch(hamil, trial, state.walkers))
    wsum = jnp.maximum(jnp.sum(state.weights), 1.0e-12)
    return jnp.sum(state.weights * e_loc) / wsum


def _calc_force_bias_det(trial: Any, walkers: Any, prop_data: PropagationData) -> Any:
    rdm = calc_rdm_batch(trial, walkers)
    rdm_sum = _spin_sum_rdm(rdm)
    return jnp.einsum("kpq,wpq->wk", prop_data.vhs, rdm_sum)


def _propagate_step_det(
    trial: Any,
    state: AFQMCState,
    prop_data: PropagationData,
    min_weight: float,
    max_weight: float,
) -> AFQMCState:
    n_walkers = state.weights.shape[0]
    nchol = prop_data.vhs.shape[0]

    key, subkey = jax.random.split(state.key)
    fields = jax.random.normal(
        subkey, (n_walkers, nchol), dtype=jnp.real(prop_data.vhs).dtype
    )

    force_bias = _calc_force_bias_det(trial, state.walkers, prop_data)
    field_shifts = -prop_data.sqrt_dt * (1.0j * force_bias - prop_data.mf_shifts)
    shifted_fields = fields - field_shifts

    shift_term = jnp.einsum("wk,k->w", shifted_fields, prop_data.mf_shifts)
    fb_term = jnp.einsum("wk,wk->w", fields, field_shifts) - 0.5 * jnp.einsum(
        "wk,wk->w", field_shifts, field_shifts
    )

    walkers_new = apply_trotter(state.walkers, shifted_fields, prop_data)
    sign_new, logov_new = calc_slov_batch(trial, walkers_new)
    # Keep dtypes stable inside lax.scan
    sign_new = sign_new.astype(state.sign.dtype)
    logov_new = logov_new.astype(state.logov.dtype)

    valid = jnp.isfinite(state.logov)
    log_ratio = jnp.where(valid, logov_new - state.logov, -jnp.inf)
    sign_ratio = jnp.where(valid, sign_new / state.sign, 0.0 + 0.0j)
    ovlp_ratio = sign_ratio * jnp.exp(log_ratio)

    imp = jnp.exp(
        -prop_data.sqrt_dt * shift_term
        + fb_term
        + prop_data.dt * (state.pop_control_shift + prop_data.h0_prop)
    ) * ovlp_ratio
    theta = jnp.angle(jnp.exp(-prop_data.sqrt_dt * shift_term) * ovlp_ratio)
    imp_ph = jnp.abs(imp) * jnp.cos(theta)
    imp_ph = jnp.where(jnp.isnan(imp_ph), 0.0, imp_ph)
    imp_ph = jnp.where(imp_ph < 0.0, 0.0, imp_ph)
    imp_ph = jnp.where(imp_ph < min_weight, 0.0, imp_ph)
    imp_ph = jnp.where(imp_ph > max_weight, 0.0, imp_ph)

    imp_ph = jnp.real(imp_ph).astype(state.weights.dtype)
    weights = (state.weights * imp_ph).astype(state.weights.dtype)
    weights = jnp.where(weights > max_weight, 0.0, weights)
    wsum = jnp.maximum(jnp.sum(weights), 1.0e-12)
    pop_control_shift = state.e_estimate - 0.1 * jnp.log(wsum / n_walkers) / prop_data.dt

    return AFQMCState(
        walkers=walkers_new,
        weights=weights,
        sign=sign_new,
        logov=logov_new,
        key=key,
        e_estimate=state.e_estimate,
        pop_control_shift=pop_control_shift,
        walker_fields=state.walker_fields,
    )


def _orthonormalize_det(
    trial: Any,
    state: AFQMCState,
    do_ortho: jnp.ndarray,
) -> AFQMCState:
    def _apply(_):
        walkers, _ = orthonormalize(state.walkers)
        sign, logov = calc_slov_batch(trial, walkers)
        return AFQMCState(
            walkers=walkers,
            weights=state.weights,
            sign=sign,
            logov=logov,
            key=state.key,
            e_estimate=state.e_estimate,
            pop_control_shift=state.pop_control_shift,
            walker_fields=state.walker_fields,
        )

    return lax.cond(do_ortho, _apply, lambda _: state, operand=None)


def _scan_steps_det(
    state: AFQMCState,
    trial: Any,
    prop_data: PropagationData,
    n_steps: int,
    min_weight: float,
    max_weight: float,
    ortho_interval: int,
    record_wsum: bool,
):
    ortho_interval_i = int(ortho_interval)

    def body(carry, _):
        st, idx = carry
        st = _propagate_step_det(trial, st, prop_data, min_weight, max_weight)
        if ortho_interval_i > 0:
            do_ortho = ((idx + 1) % ortho_interval_i) == 0
            st = _orthonormalize_det(trial, st, do_ortho)
        wsum = jnp.sum(st.weights)
        return (st, idx + 1), wsum

    (state, _), wsum_hist = lax.scan(body, (state, 0), None, length=n_steps)
    if record_wsum:
        return state, wsum_hist
    return state, jnp.zeros((0,), dtype=wsum_hist.dtype)


def afqmc_energy(
    hamil: Hamiltonian,
    trial: Optional[Any] = None,
    *,
    cfg: Optional[AFQMCConfig] = None,
    logger: Optional[logging.Logger] = None,
    return_state: bool = False,
    return_blocks: bool = False,
):
    cfg = cfg or AFQMCConfig()
    logger = _get_logger(logger)
    start = time.time()

    if trial is None:
        trial = hamil.wfn0

    prop_data = build_propagation_data(hamil, trial, cfg.dt)
    key = jax.random.PRNGKey(cfg.seed)
    walkers, key = init_walkers(trial, cfg.n_walkers, key, noise=cfg.init_noise)
    if _is_custom_trial(trial) and hasattr(trial, "bind_walkers"):
        key, trial_key = jax.random.split(key)
        trial.bind_walkers(walkers, key=trial_key, reinit=True)
    if not _is_custom_trial(trial):
        # Ensure stable dtypes for jit/scan
        w_up, w_dn = walkers
        walkers = (w_up.astype(jnp.complex128), w_dn.astype(jnp.complex128))
    sign, logov = calc_slov_batch(trial, walkers)
    sign = sign.astype(jnp.complex128)
    logov = logov.astype(jnp.float64)
    weights = jnp.ones((cfg.n_walkers,), dtype=jnp.float64)

    e_samples = jnp.real(calc_local_energy_batch(hamil, trial, walkers))
    e_estimate = jnp.sum(e_samples) / cfg.n_walkers

    state = AFQMCState(
        walkers=walkers,
        weights=weights,
        sign=sign,
        logov=logov,
        key=key,
        e_estimate=e_estimate,
        pop_control_shift=e_estimate,
        walker_fields=(
            trial.walker_fields
            if _is_custom_trial(trial)
            and hasattr(trial, "walker_fields")
            and getattr(trial, "walker_fields") is not None
            else jnp.zeros((0, 0, 0), dtype=jnp.float64)
        ),
    )

    logger.info(
        "AFQMC init: dt=%.4g walkers=%d blocks=%d prop_steps=%d",
        cfg.dt,
        cfg.n_walkers,
        cfg.n_blocks,
        cfg.n_prop_steps,
    )
    logger.info("Init e_est=%.12f elapsed=%.2fs", float(e_estimate), time.time() - start)

    use_jit = not _is_custom_trial(trial)
    if use_jit:
        record_eq = cfg.log_interval > 0
        eq_scan = jax.jit(
            lambda st: _scan_steps_det(
                st,
                trial,
                prop_data,
                cfg.n_eq_steps,
                cfg.min_weight,
                cfg.max_weight,
                cfg.ortho_interval,
                record_eq,
            )
        )
        state, wsum_hist = eq_scan(state)
        if cfg.log_interval > 0:
            for step in range(cfg.log_interval - 1, cfg.n_eq_steps, cfg.log_interval):
                wsum = float(wsum_hist[step])
                logger.info(
                    "Eql %d/%d wsum=%.3e e_est=%.12f elapsed=%.2fs",
                    step + 1,
                    cfg.n_eq_steps,
                    wsum,
                    float(state.e_estimate),
                    time.time() - start,
                )
    else:
        pure_err_logged = False
        for step in range(cfg.n_eq_steps):
            state = _propagate_step(hamil, trial, state, prop_data, cfg)
            state = maybe_orthonormalize(trial, state, step, cfg.ortho_interval)
            if cfg.log_interval > 0 and (
                (step + 1) % cfg.log_interval == 0 or step + 1 == cfg.n_eq_steps
            ):
                wsum = float(jnp.sum(state.weights))
                e_mix = float(_calc_mixed_energy(hamil, trial, state))
                e_trial_msg = ""
                pure_pairs = int(getattr(trial, "pure_eval_pairs", 1))
                if pure_pairs > 0 and hasattr(trial, "calc_pure_energy_inference"):
                    try:
                        e_trial = float(jnp.real(trial.calc_pure_energy_inference(hamil)))
                        e_trial_msg = f" e_trial_pure={e_trial:.12f}"
                    except Exception as exc:
                        if (not pure_err_logged) and logger is not None:
                            logger.warning(
                                "trial pure-energy diagnostic failed; disabling pure log in AFQMC eql. error=%s",
                                str(exc),
                            )
                            pure_err_logged = True
                        if hasattr(trial, "pure_eval_pairs"):
                            trial.pure_eval_pairs = 0
                logger.info(
                    "Eql %d/%d wsum=%.3e e_mix=%.12f e_est=%.12f%s elapsed=%.2fs",
                    step + 1,
                    cfg.n_eq_steps,
                    wsum,
                    e_mix,
                    float(state.e_estimate),
                    e_trial_msg,
                    time.time() - start,
                )

    block_energies = []
    block_scan = None
    if use_jit:
        block_scan = jax.jit(
            lambda st: _scan_steps_det(
                st,
                trial,
                prop_data,
                cfg.n_prop_steps,
                cfg.min_weight,
                cfg.max_weight,
                cfg.ortho_interval,
                False,
            )
        )

    for blk in range(cfg.n_blocks):
        if use_jit:
            state, _ = block_scan(state)
        else:
            for step in range(cfg.n_prop_steps):
                state = _propagate_step(hamil, trial, state, prop_data, cfg)
                state = maybe_orthonormalize(trial, state, step, cfg.ortho_interval)

        e_loc = jnp.real(calc_local_energy_batch(hamil, trial, state.walkers))
        clip = jnp.sqrt(2.0 / cfg.dt)
        e_loc = jnp.where(
            jnp.abs(e_loc - state.e_estimate) > clip, state.e_estimate, e_loc
        )
        wsum = jnp.maximum(jnp.sum(state.weights), 1.0e-12)
        block_energy = jnp.sum(state.weights * e_loc) / wsum
        block_energies.append(block_energy)

        state = AFQMCState(
            walkers=state.walkers,
            weights=state.weights,
            sign=state.sign,
            logov=state.logov,
            key=state.key,
            e_estimate=0.9 * state.e_estimate + 0.1 * block_energy,
            pop_control_shift=state.pop_control_shift,
            walker_fields=state.walker_fields,
        )

        if cfg.resample:
            state.key, subkey = jax.random.split(state.key)
            state = stochastic_reconfiguration(trial, state, subkey)

        if cfg.log_interval > 0 and (
            (blk + 1) % cfg.log_interval == 0 or blk + 1 == cfg.n_blocks
        ):
            logger.info(
                "Block %d/%d e_blk=%.12f e_est=%.12f wsum=%.3e elapsed=%.2fs",
                blk + 1,
                cfg.n_blocks,
                float(block_energy),
                float(state.e_estimate),
                float(jnp.sum(state.weights)),
                time.time() - start,
            )

    blocks = jnp.array(block_energies)
    e_mean = jnp.mean(blocks)
    e_err = jnp.std(blocks, ddof=1) / jnp.sqrt(blocks.size) if blocks.size > 1 else 0.0

    logger.info(
        "AFQMC done: E=%.12f +/- %.3e elapsed=%.2fs",
        float(e_mean),
        float(e_err),
        time.time() - start,
    )

    if return_state and return_blocks:
        return e_mean, e_err, blocks, state
    if return_state:
        return e_mean, e_err, state
    if return_blocks:
        return e_mean, e_err, blocks
    return e_mean, e_err


def afqmc_energy_from_pickle(
    hamil_path: str,
    *,
    cfg: Optional[AFQMCConfig] = None,
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
    cfg: Optional[AFQMCConfig] = None,
    logger: Optional[logging.Logger] = None,
    n_walkers: Optional[int] = None,
    n_samples: int = 20,
    burn_in: int = 1000,
    sampler_name: str = "hmc",
    sampler_kwargs: Optional[dict] = None,
    sampling_target: str = "walker_overlap",
    logdens_floor: float = -60.0,
    refresh_interval: int = 0,
    replace_per_refresh: int = 1,
    sample_update_steps: int = 1,
    init_walkers_from_trial: bool = False,
    init_walkers_burn_in: int = 0,
    pure_eval_pairs: int = 1000,
    max_prop: Optional[Any] = None,
    seed: int = 0,
    return_state: bool = False,
    return_blocks: bool = False,
):
    """Run AFQMC by building VAFQMCTrial from checkpoint + hparams."""
    cfg = cfg or AFQMCConfig()
    if n_walkers is not None:
        cfg = AFQMCConfig(**{**cfg.__dict__, "n_walkers": int(n_walkers)})

    hamil = load_hamiltonian(hamil_path)
    trial = VAFQMCTrial.from_hparams_checkpoint(
        hamil,
        checkpoint_path,
        hparams_path=hparams_path,
        n_samples=n_samples,
        burn_in=burn_in,
        sampler_name=sampler_name,
        sampler_kwargs=sampler_kwargs,
        sampling_target=sampling_target,
        logdens_floor=logdens_floor,
        refresh_interval=refresh_interval,
        replace_per_refresh=replace_per_refresh,
        sample_update_steps=sample_update_steps,
        init_walkers_from_trial=init_walkers_from_trial,
        init_walkers_burn_in=init_walkers_burn_in,
        pure_eval_pairs=pure_eval_pairs,
        max_prop=max_prop,
        seed=seed,
    )
    return afqmc_energy(
        hamil,
        trial=trial,
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
