"""Single-determinant AFQMC driver internals."""

from __future__ import annotations

import logging
import time
from typing import Any

import jax
from jax import lax, numpy as jnp

from ...hamiltonian import Hamiltonian
from ...propagator import orthonormalize
from ..afqmc_config import AFQMCConfig
from ..afqmc_utils import (
    PropagationData,
    apply_trotter,
    build_propagation_data,
    calc_local_energy_batch,
    calc_rdm_batch,
    calc_slov_batch,
    _spin_sum_rdm,
)
from ..walker import AFQMCState, init_walkers, stochastic_reconfiguration


def _ensure_complex_walkers(walkers: Any) -> Any:
    if isinstance(walkers, tuple) and len(walkers) == 2:
        w_up, w_dn = walkers
        return (w_up.astype(jnp.complex128), w_dn.astype(jnp.complex128))
    return walkers


def _log_init(logger: logging.Logger, cfg: AFQMCConfig, state: AFQMCState, start: float) -> None:
    logger.info(
        "AFQMC init: dt=%.4g walkers=%d blocks=%d prop_steps=%d meas_samples=%d",
        cfg.dt,
        cfg.n_walkers,
        cfg.n_blocks,
        cfg.n_prop_steps,
        max(int(getattr(cfg, "n_measure_samples", 1)), 1),
    )
    logger.info("Init e_est=%.12f elapsed=%.2fs", float(state.e_estimate), time.time() - start)


def _update_e_estimate(state: AFQMCState, block_energy: Any) -> AFQMCState:
    return AFQMCState(
        walkers=state.walkers,
        weights=state.weights,
        sign=state.sign,
        logov=state.logov,
        key=state.key,
        e_estimate=0.9 * state.e_estimate + 0.1 * block_energy,
        pop_control_shift=state.pop_control_shift,
        walker_fields=state.walker_fields,
        trial_state=state.trial_state,
    )


def _finalize_outputs(
    logger: logging.Logger,
    start: float,
    block_energies: list[Any],
    *,
    state: AFQMCState,
    return_state: bool,
    return_blocks: bool,
):
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
        trial_state=state.trial_state,
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
            trial_state=state.trial_state,
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


def _build_initial_state_det(
    hamil: Hamiltonian,
    trial: Any,
    cfg: AFQMCConfig,
) -> tuple[PropagationData, AFQMCState]:
    prop_data = build_propagation_data(hamil, trial, cfg.dt)
    key = jax.random.PRNGKey(cfg.seed)
    walkers, key = init_walkers(trial, cfg.n_walkers, key, noise=cfg.init_noise)
    walkers = _ensure_complex_walkers(walkers)

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
        walker_fields=jnp.zeros((0, 0, 0), dtype=jnp.float64),
        trial_state=None,
    )
    return prop_data, state


def _run_det_equilibration(
    state: AFQMCState,
    trial: Any,
    prop_data: PropagationData,
    cfg: AFQMCConfig,
    logger: logging.Logger,
    start: float,
) -> AFQMCState:
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
            logger.info(
                "Eql %d/%d wsum=%.3e e_est=%.12f elapsed=%.2fs",
                step + 1,
                cfg.n_eq_steps,
                float(wsum_hist[step]),
                float(state.e_estimate),
                time.time() - start,
            )
    return state


def _measure_block_energy_det(
    hamil: Hamiltonian,
    trial: Any,
    state: AFQMCState,
    cfg: AFQMCConfig,
) -> Any:
    e_loc = jnp.real(calc_local_energy_batch(hamil, trial, state.walkers))
    clip = jnp.sqrt(2.0 / cfg.dt)
    e_loc = jnp.where(jnp.abs(e_loc - state.e_estimate) > clip, state.e_estimate, e_loc)
    wsum = jnp.maximum(jnp.sum(state.weights), 1.0e-12)
    return jnp.sum(state.weights * e_loc) / wsum


def run_afqmc_det(
    hamil: Hamiltonian,
    trial: Any,
    cfg: AFQMCConfig,
    logger: logging.Logger,
    start: float,
    *,
    return_state: bool,
    return_blocks: bool,
):
    prop_data, state = _build_initial_state_det(hamil, trial, cfg)
    _log_init(logger, cfg, state, start)

    state = _run_det_equilibration(state, trial, prop_data, cfg, logger, start)

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

    block_energies: list[Any] = []
    for blk in range(cfg.n_blocks):
        state, _ = block_scan(state)
        block_energy = _measure_block_energy_det(hamil, trial, state, cfg)
        block_energies.append(block_energy)
        state = _update_e_estimate(state, block_energy)

        if cfg.resample:
            key_new, subkey = jax.random.split(state.key)
            state = AFQMCState(
                walkers=state.walkers,
                weights=state.weights,
                sign=state.sign,
                logov=state.logov,
                key=key_new,
                e_estimate=state.e_estimate,
                pop_control_shift=state.pop_control_shift,
                walker_fields=state.walker_fields,
                trial_state=state.trial_state,
            )
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

    return _finalize_outputs(
        logger,
        start,
        block_energies,
        state=state,
        return_state=return_state,
        return_blocks=return_blocks,
    )


__all__ = ["run_afqmc_det"]
