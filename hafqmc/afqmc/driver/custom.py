"""Custom-trial AFQMC driver internals."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    calc_force_bias,
    calc_local_energy_batch,
    calc_slov_batch,
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


def _propagate_step_custom(
    hamil: Hamiltonian,
    trial: Any,
    state: AFQMCState,
    prop_data: PropagationData,
    cfg: AFQMCConfig,
) -> AFQMCState:
    n_walkers = state.weights.shape[0]
    nchol = prop_data.vhs.shape[0]
    trial_state = state.trial_state

    key, subkey = jax.random.split(state.key)
    fields = jax.random.normal(
        subkey, (n_walkers, nchol), dtype=jnp.real(prop_data.vhs).dtype
    )

    if trial_state is not None and hasattr(trial, "calc_force_bias_state"):
        force_bias = trial.calc_force_bias_state(state.walkers, prop_data, trial_state)
    else:
        force_bias = calc_force_bias(hamil, trial, state.walkers, prop_data)
    field_shifts = -prop_data.sqrt_dt * (1.0j * force_bias - prop_data.mf_shifts)
    shifted_fields = fields - field_shifts

    shift_term = jnp.einsum("wk,k->w", shifted_fields, prop_data.mf_shifts)
    fb_term = jnp.einsum("wk,wk->w", fields, field_shifts) - 0.5 * jnp.einsum(
        "wk,wk->w", field_shifts, field_shifts
    )

    walkers_new = apply_trotter(state.walkers, shifted_fields, prop_data)
    if trial_state is not None and hasattr(trial, "calc_slov_state"):
        sign_prop, logov_prop = trial.calc_slov_state(walkers_new, trial_state)
    else:
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

    sign_new, logov_new = sign_prop, logov_prop
    walker_fields = state.walker_fields
    if trial_state is not None and hasattr(trial, "update_tethered_samples_state"):
        trial_state, handoff, sign_new, logov_new = trial.update_tethered_samples_state(
            walkers_new,
            trial_state,
            old_sign=sign_prop,
            old_logov=logov_prop,
        )
        if isinstance(trial_state, dict) and ("walker_fields" in trial_state):
            walker_fields = trial_state["walker_fields"]
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
        trial_state=trial_state,
    )


def _calc_mixed_energy_custom(hamil: Hamiltonian, trial: Any, state: AFQMCState) -> Any:
    if state.trial_state is not None and hasattr(trial, "calc_local_energy_state"):
        e_loc = jnp.real(trial.calc_local_energy_state(hamil, state.walkers, state.trial_state))
    else:
        e_loc = jnp.real(calc_local_energy_batch(hamil, trial, state.walkers))
    wsum = jnp.maximum(jnp.sum(state.weights), 1.0e-12)
    return jnp.sum(state.weights * e_loc) / wsum


def _calc_block_energy_once_custom(
    hamil: Hamiltonian,
    trial: Any,
    state: AFQMCState,
    cfg: AFQMCConfig,
) -> Any:
    if state.trial_state is not None and hasattr(trial, "calc_local_energy_state"):
        e_loc = jnp.real(trial.calc_local_energy_state(hamil, state.walkers, state.trial_state))
    else:
        e_loc = jnp.real(calc_local_energy_batch(hamil, trial, state.walkers))
    clip = jnp.sqrt(2.0 / cfg.dt)
    e_loc = jnp.where(jnp.abs(e_loc - state.e_estimate) > clip, state.e_estimate, e_loc)
    wsum = jnp.maximum(jnp.sum(state.weights), 1.0e-12)
    return jnp.sum(state.weights * e_loc) / wsum


def _orthonormalize_custom(
    trial: Any,
    state: AFQMCState,
    do_ortho: jnp.ndarray,
) -> AFQMCState:
    def _apply(_):
        trial_state = state.trial_state
        if hasattr(trial, "orthonormalize_walkers"):
            walkers = trial.orthonormalize_walkers(state.walkers)
            if trial_state is not None and hasattr(trial, "on_walkers_updated_state"):
                trial_state = trial.on_walkers_updated_state(walkers, trial_state)
            elif hasattr(trial, "on_walkers_updated"):
                trial.on_walkers_updated(walkers)
                if hasattr(trial, "export_runtime_state"):
                    trial_state = trial.export_runtime_state()
        else:
            walkers, _ = orthonormalize(state.walkers)

        if trial_state is not None and hasattr(trial, "calc_slov_state"):
            sign, logov = trial.calc_slov_state(walkers, trial_state)
        else:
            sign, logov = calc_slov_batch(trial, walkers)
        walker_fields = (
            trial_state.get("walker_fields", state.walker_fields)
            if isinstance(trial_state, dict)
            else state.walker_fields
        )
        return AFQMCState(
            walkers=walkers,
            weights=state.weights,
            sign=sign,
            logov=logov,
            key=state.key,
            e_estimate=state.e_estimate,
            pop_control_shift=state.pop_control_shift,
            walker_fields=walker_fields,
            trial_state=trial_state,
        )

    return lax.cond(do_ortho, _apply, lambda _: state, operand=None)


def _scan_steps_custom(
    state: AFQMCState,
    hamil: Hamiltonian,
    trial: Any,
    prop_data: PropagationData,
    cfg: AFQMCConfig,
    n_steps: int,
    record_wsum: bool,
):
    ortho_interval_i = int(cfg.ortho_interval)

    def body(carry, _):
        st, idx = carry
        st = _propagate_step_custom(hamil, trial, st, prop_data, cfg)
        if ortho_interval_i > 0:
            do_ortho = ((idx + 1) % ortho_interval_i) == 0
            st = _orthonormalize_custom(trial, st, do_ortho)
        wsum = jnp.sum(st.weights)
        return (st, idx + 1), wsum

    (state, _), wsum_hist = lax.scan(body, (state, 0), None, length=n_steps)
    if record_wsum:
        return state, wsum_hist
    return state, jnp.zeros((0,), dtype=wsum_hist.dtype)


@dataclass
class _CustomScanRunner:
    """Run custom-trial propagation with scan+jit and one-way Python fallback."""

    hamil: Hamiltonian
    trial: Any
    prop_data: PropagationData
    cfg: AFQMCConfig
    logger: logging.Logger
    enabled: bool = True
    cache: dict[int, Any] = field(default_factory=dict)

    def _get_scan(self, n_steps: int):
        n_steps_i = int(n_steps)
        fn = self.cache.get(n_steps_i)
        if fn is None:
            fn = jax.jit(
                lambda st: _scan_steps_custom(
                    st,
                    self.hamil,
                    self.trial,
                    self.prop_data,
                    self.cfg,
                    n_steps_i,
                    False,
                )
            )
            self.cache[n_steps_i] = fn
        return fn

    def run_steps(self, state: AFQMCState, n_steps: int, label: str) -> AFQMCState:
        n_steps_i = int(n_steps)
        if n_steps_i <= 0:
            return state

        if self.enabled:
            try:
                state, _ = self._get_scan(n_steps_i)(state)
                return state
            except Exception as exc:
                self.logger.warning(
                    "Custom-trial %s scan-jit failed; fallback to python loop. error=%s",
                    label,
                    str(exc),
                )
                self.enabled = False

        for step in range(n_steps_i):
            state = _propagate_step_custom(self.hamil, self.trial, state, self.prop_data, self.cfg)
            if self.cfg.ortho_interval > 0 and (step + 1) % self.cfg.ortho_interval == 0:
                state = _orthonormalize_custom(self.trial, state, jnp.asarray(True))
        return state


def _bind_trial_runtime_state_custom(trial: Any, walkers: Any, key: Any) -> tuple[Any, Any]:
    trial_state = None
    if hasattr(trial, "bind_walkers"):
        if hasattr(trial, "init_runtime_state"):
            key, trial_key = jax.random.split(key)
            trial_state = trial.init_runtime_state(walkers, key=trial_key, reinit=True)
        else:
            key, trial_key = jax.random.split(key)
            trial.bind_walkers(walkers, key=trial_key, reinit=True)
            if hasattr(trial, "export_runtime_state"):
                trial_state = trial.export_runtime_state()
    return trial_state, key


def _calc_trial_overlap_custom(trial: Any, walkers: Any, trial_state: Any):
    if trial_state is not None and hasattr(trial, "calc_slov_state"):
        return trial.calc_slov_state(walkers, trial_state)
    return calc_slov_batch(trial, walkers)


def _calc_trial_local_energy_custom(
    hamil: Hamiltonian,
    trial: Any,
    walkers: Any,
    trial_state: Any,
):
    if trial_state is not None and hasattr(trial, "calc_local_energy_state"):
        return jnp.real(trial.calc_local_energy_state(hamil, walkers, trial_state))
    return jnp.real(calc_local_energy_batch(hamil, trial, walkers))


def _initial_walker_fields_custom(trial: Any, trial_state: Any):
    if isinstance(trial_state, dict) and ("walker_fields" in trial_state):
        return trial_state["walker_fields"]
    if hasattr(trial, "walker_fields") and getattr(trial, "walker_fields") is not None:
        return trial.walker_fields
    return jnp.zeros((0, 0, 0), dtype=jnp.float64)


def _build_initial_state_custom(
    hamil: Hamiltonian,
    trial: Any,
    cfg: AFQMCConfig,
) -> tuple[PropagationData, AFQMCState]:
    prop_data = build_propagation_data(hamil, trial, cfg.dt)
    key = jax.random.PRNGKey(cfg.seed)
    walkers, key = init_walkers(trial, cfg.n_walkers, key, noise=cfg.init_noise)
    walkers = _ensure_complex_walkers(walkers)

    trial_state, key = _bind_trial_runtime_state_custom(trial, walkers, key)
    sign, logov = _calc_trial_overlap_custom(trial, walkers, trial_state)
    sign = sign.astype(jnp.complex128)
    logov = logov.astype(jnp.float64)
    weights = jnp.ones((cfg.n_walkers,), dtype=jnp.float64)

    e_samples = _calc_trial_local_energy_custom(hamil, trial, walkers, trial_state)
    e_estimate = jnp.sum(e_samples) / cfg.n_walkers

    state = AFQMCState(
        walkers=walkers,
        weights=weights,
        sign=sign,
        logov=logov,
        key=key,
        e_estimate=e_estimate,
        pop_control_shift=e_estimate,
        walker_fields=_initial_walker_fields_custom(trial, trial_state),
        trial_state=trial_state,
    )
    return prop_data, state


def _run_custom_equilibration(
    state: AFQMCState,
    runner: _CustomScanRunner,
    hamil: Hamiltonian,
    trial: Any,
    cfg: AFQMCConfig,
    logger: logging.Logger,
    start: float,
) -> AFQMCState:
    if cfg.log_interval <= 0:
        return runner.run_steps(state, cfg.n_eq_steps, "eql")

    eq_done = 0
    eq_chunk = max(int(cfg.log_interval), 1)
    while eq_done < cfg.n_eq_steps:
        nrun = min(eq_chunk, cfg.n_eq_steps - eq_done)
        state = runner.run_steps(state, nrun, "eql")
        eq_done += nrun
        logger.info(
            "Eql %d/%d wsum=%.3e e_mix=%.12f e_est=%.12f elapsed=%.2fs",
            eq_done,
            cfg.n_eq_steps,
            float(jnp.sum(state.weights)),
            float(_calc_mixed_energy_custom(hamil, trial, state)),
            float(state.e_estimate),
            time.time() - start,
        )
    return state


def _measure_block_energy_custom(
    hamil: Hamiltonian,
    trial: Any,
    state: AFQMCState,
    cfg: AFQMCConfig,
) -> tuple[AFQMCState, Any]:
    n_meas = max(int(getattr(cfg, "n_measure_samples", 1)), 1)
    if state.trial_state is not None and hasattr(trial, "measure_block_energy_state"):
        trial_state, block_energy, sign_cur, logov_cur = trial.measure_block_energy_state(
            hamil,
            state.walkers,
            state.weights,
            state.e_estimate,
            cfg.dt,
            n_meas,
            state.trial_state,
        )
        walker_fields = (
            trial_state.get("walker_fields", state.walker_fields)
            if isinstance(trial_state, dict)
            else state.walker_fields
        )
        state = AFQMCState(
            walkers=state.walkers,
            weights=state.weights,
            sign=sign_cur,
            logov=logov_cur,
            key=state.key,
            e_estimate=state.e_estimate,
            pop_control_shift=state.pop_control_shift,
            walker_fields=walker_fields,
            trial_state=trial_state,
        )
        return state, block_energy

    return state, _calc_block_energy_once_custom(hamil, trial, state, cfg)


def run_afqmc_custom(
    hamil: Hamiltonian,
    trial: Any,
    cfg: AFQMCConfig,
    logger: logging.Logger,
    start: float,
    *,
    return_state: bool,
    return_blocks: bool,
):
    prop_data, state = _build_initial_state_custom(hamil, trial, cfg)
    _log_init(logger, cfg, state, start)

    custom_runner = _CustomScanRunner(hamil, trial, prop_data, cfg, logger)
    state = _run_custom_equilibration(state, custom_runner, hamil, trial, cfg, logger, start)

    block_energies: list[Any] = []
    for blk in range(cfg.n_blocks):
        state = custom_runner.run_steps(state, cfg.n_prop_steps, "block")
        state, block_energy = _measure_block_energy_custom(hamil, trial, state, cfg)
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


__all__ = ["run_afqmc_custom"]
