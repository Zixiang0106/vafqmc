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
    analyze_energy_blocks,
    apply_trotter,
    build_propagation_data,
    calc_local_energy_batch,
    calc_rdm_batch,
    calc_slov_batch,
    _spin_sum_rdm,
)
from ..utils.visualization import build_energy_visualizer
from ..walker import AFQMCState, init_walkers, stochastic_reconfiguration


def _ensure_complex_walkers(walkers: Any) -> Any:
    if isinstance(walkers, tuple) and len(walkers) == 2:
        w_up, w_dn = walkers
        return (w_up.astype(jnp.complex128), w_dn.astype(jnp.complex128))
    return walkers


def _log_init(logger: logging.Logger, cfg: AFQMCConfig, state: AFQMCState, start: float) -> None:
    logger.info(
        "AFQMC init: dt=%.4g walkers=%d blocks=%d sr_blocks=%d ene_blocks=%d prop_steps=%d meas_samples=%d pop_freq=%d",
        cfg.dt,
        cfg.n_walkers,
        cfg.n_blocks,
        max(int(getattr(cfg, "n_sr_blocks", 1)), 1),
        max(int(getattr(cfg, "n_ene_blocks", 1)), 1),
        cfg.n_prop_steps,
        max(int(getattr(cfg, "n_measure_samples", 1)), 1),
        int(getattr(cfg, "pop_control_freq", 0)),
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


def _relax_pop_control_shift(state: AFQMCState, block_energy: Any) -> AFQMCState:
    return AFQMCState(
        walkers=state.walkers,
        weights=state.weights,
        sign=state.sign,
        logov=state.logov,
        key=state.key,
        e_estimate=state.e_estimate,
        pop_control_shift=0.9 * state.pop_control_shift + 0.1 * block_energy,
        walker_fields=state.walker_fields,
        trial_state=state.trial_state,
    )


def _finalize_outputs(
    logger: logging.Logger,
    start: float,
    block_energies: list[Any],
    block_weights: list[Any],
    *,
    state: AFQMCState,
    visualizer: Any,
    return_state: bool,
    return_blocks: bool,
):
    blocks = jnp.array(block_energies)
    weights = jnp.array(block_weights)
    e_mean, e_err, n_outliers = analyze_energy_blocks(blocks, weights)

    logger.info(
        "AFQMC done: E=%.12f +/- %.3e outliers=%d elapsed=%.2fs",
        float(e_mean),
        float(e_err),
        int(n_outliers),
        time.time() - start,
    )
    visualizer.finalize(float(e_mean), float(e_err))

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
        + prop_data.dt * (state.pop_control_shift - prop_data.h0_prop)
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
    step_start: int,
    record_wsum: bool,
):
    ortho_interval_i = int(ortho_interval)
    step_start_i = jnp.asarray(step_start, dtype=jnp.int32)

    def body(carry, _):
        st, idx = carry
        st = _propagate_step_det(trial, st, prop_data, min_weight, max_weight)
        if ortho_interval_i > 0:
            do_ortho = ((step_start_i + idx + 1) % ortho_interval_i) == 0
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


def _resample_state_det(trial: Any, state: AFQMCState) -> AFQMCState:
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
    return stochastic_reconfiguration(trial, state, subkey)


def _run_steps_with_pop_control_det(
    state: AFQMCState,
    run_chunk: Any,
    *,
    n_steps: int,
    step_counter: int,
    pop_freq: int,
    record_wsum: bool,
    do_resample: bool,
    trial: Any,
) -> tuple[AFQMCState, Any, int]:
    steps_done = 0
    local_step = 0
    last_wsum = jnp.sum(state.weights)
    while steps_done < n_steps:
        if pop_freq > 0:
            to_pop = pop_freq - (step_counter % pop_freq)
            nrun = min(n_steps - steps_done, to_pop)
        else:
            nrun = n_steps - steps_done
        state, wsum_hist = run_chunk(state, nrun, local_step, record_wsum)
        step_counter += nrun
        steps_done += nrun
        local_step += nrun
        if record_wsum and wsum_hist.size > 0:
            last_wsum = wsum_hist[-1]
        elif not record_wsum:
            last_wsum = jnp.sum(state.weights)

        if do_resample and pop_freq > 0 and step_counter % pop_freq == 0:
            state = _resample_state_det(trial, state)

    return state, last_wsum, step_counter


def _run_det_equilibration(
    state: AFQMCState,
    hamil: Hamiltonian,
    trial: Any,
    prop_data: PropagationData,
    cfg: AFQMCConfig,
    logger: logging.Logger,
    start: float,
) -> AFQMCState:
    if cfg.n_eq_steps <= 0:
        return state

    cache: dict[tuple[int, bool], Any] = {}

    def run_chunk(st: AFQMCState, n_steps: int, step_start: int, record_wsum: bool):
        key = (int(n_steps), bool(record_wsum))
        fn = cache.get(key)
        if fn is None:
            fn = jax.jit(
                lambda s, step0: _scan_steps_det(
                    s,
                    trial,
                    prop_data,
                    key[0],
                    cfg.min_weight,
                    cfg.max_weight,
                    cfg.ortho_interval,
                    step0,
                    key[1],
                )
            )
            cache[key] = fn
        return fn(st, int(step_start))

    pop_freq = int(getattr(cfg, "pop_control_freq", 0))
    step_counter = 0
    eq_done = 0
    log_enabled = bool(getattr(cfg, "log_enabled", True))
    eq_interval = int(getattr(cfg, "log_equil_interval", 0))
    eq_n_print = max(int(getattr(cfg, "log_equil_n_print", 5)), 0)
    if log_enabled and eq_interval > 0:
        log_eq = True
        eq_chunk = max(eq_interval, 1)
    elif log_enabled and eq_n_print > 0:
        log_eq = True
        eq_chunk = max(int(cfg.n_eq_steps) // eq_n_print, 1)
    else:
        log_eq = False
        eq_chunk = int(cfg.n_eq_steps)
    if log_eq:
        logger.info(
            "Equilibration sweeps: steps=%d, print_every=%d",
            int(cfg.n_eq_steps),
            int(eq_chunk),
        )
    while eq_done < cfg.n_eq_steps:
        nrun = min(eq_chunk, cfg.n_eq_steps - eq_done)
        record_wsum = log_eq
        if pop_freq > 0:
            state, wsum_last, step_counter = _run_steps_with_pop_control_det(
                state,
                run_chunk,
                n_steps=nrun,
                step_counter=step_counter,
                pop_freq=pop_freq,
                record_wsum=record_wsum,
                do_resample=bool(cfg.resample),
                trial=trial,
            )
        else:
            state, wsum_hist = run_chunk(state, nrun, 0, record_wsum)
            wsum_last = wsum_hist[-1] if record_wsum and wsum_hist.size > 0 else jnp.sum(state.weights)
        eq_done += nrun

        e_mix = _measure_block_energy_det(hamil, trial, state, cfg)
        state = _update_e_estimate(state, e_mix)
        state = _relax_pop_control_shift(state, e_mix)

        if cfg.resample and pop_freq <= 0:
            state = _resample_state_det(trial, state)

        if log_eq:
            logger.info(
                "Eql %d/%d wsum=%.3e e_mix=%.12f e_est=%.12f elapsed=%.2fs",
                eq_done,
                cfg.n_eq_steps,
                float(wsum_last),
                float(e_mix),
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

    state = _run_det_equilibration(state, hamil, trial, prop_data, cfg, logger, start)
    visualizer = build_energy_visualizer(
        enabled=bool(getattr(cfg, "visualization", False)),
        logger=logger,
        title="AFQMC Live Energy (single_det)",
        refresh_every=int(getattr(cfg, "visualization_refresh_every", 1)),
        show=bool(getattr(cfg, "visualization_show", True)),
        save_path=getattr(cfg, "visualization_save_path", None),
    )

    block_scan = jax.jit(
        lambda st: _scan_steps_det(
            st,
            trial,
            prop_data,
            cfg.n_prop_steps,
            cfg.min_weight,
            cfg.max_weight,
            cfg.ortho_interval,
            0,
            False,
        )
    )
    block_scan_cache: dict[int, Any] = {}

    def run_block_steps(st: AFQMCState, n_steps: int, step_start: int):
        n_steps_i = int(n_steps)
        if n_steps_i == int(cfg.n_prop_steps) and int(step_start) == 0:
            return block_scan(st)
        fn = block_scan_cache.get(n_steps_i)
        if fn is None:
            fn = jax.jit(
                lambda s, step0: _scan_steps_det(
                    s,
                    trial,
                    prop_data,
                    n_steps_i,
                    cfg.min_weight,
                    cfg.max_weight,
                    cfg.ortho_interval,
                    step0,
                    False,
                )
            )
            block_scan_cache[n_steps_i] = fn
        return fn(st, int(step_start))

    block_energies: list[Any] = []
    block_weights: list[Any] = []
    n_sr_blocks = max(int(getattr(cfg, "n_sr_blocks", 1)), 1)
    n_ene_blocks = max(int(getattr(cfg, "n_ene_blocks", 1)), 1)
    pop_freq = int(getattr(cfg, "pop_control_freq", 0))
    log_enabled = bool(getattr(cfg, "log_enabled", True))
    block_log_interval = int(getattr(cfg, "log_block_interval", 1))
    step_counter = 0

    for blk in range(cfg.n_blocks):
        e_num = jnp.asarray(0.0, dtype=jnp.float64)
        e_den = jnp.asarray(0.0, dtype=jnp.float64)

        for _ in range(n_sr_blocks):
            for _ in range(n_ene_blocks):
                if cfg.resample and pop_freq > 0:
                    steps_left = int(cfg.n_prop_steps)
                    local_step = 0
                    while steps_left > 0:
                        to_pop = pop_freq - (step_counter % pop_freq)
                        nrun = min(steps_left, to_pop)
                        state, _ = run_block_steps(state, nrun, local_step)
                        step_counter += nrun
                        steps_left -= nrun
                        local_step += nrun
                        if step_counter % pop_freq == 0:
                            state = _resample_state_det(trial, state)
                else:
                    state, _ = run_block_steps(state, cfg.n_prop_steps, 0)
                    step_counter += int(cfg.n_prop_steps)
                e_i = _measure_block_energy_det(hamil, trial, state, cfg)
                w_i = jnp.maximum(jnp.sum(state.weights), 1.0e-12)
                e_num = e_num + w_i * e_i
                e_den = e_den + w_i
                state = _relax_pop_control_shift(state, e_i)

            if cfg.resample and pop_freq <= 0:
                state = _resample_state_det(trial, state)

        block_energy = e_num / jnp.maximum(e_den, 1.0e-12)
        block_energies.append(block_energy)
        block_weights.append(e_den)
        state = _update_e_estimate(state, block_energy)
        visualizer.update(blk + 1, float(block_energy))

        if log_enabled and block_log_interval > 0 and (
            (blk + 1) % block_log_interval == 0 or blk + 1 == cfg.n_blocks
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
        block_weights,
        state=state,
        visualizer=visualizer,
        return_state=return_state,
        return_blocks=return_blocks,
    )


__all__ = ["run_afqmc_det"]
