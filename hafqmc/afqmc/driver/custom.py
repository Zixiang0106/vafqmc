"""Custom-trial AFQMC driver internals."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import time
from pathlib import Path
from typing import Any, TextIO

import jax
from jax import lax, numpy as jnp
from jax.tree_util import tree_leaves, tree_map

from ...hamiltonian import Hamiltonian, calc_slov
from ...propagator import orthonormalize
from ...sampler import choose_sampler_maker, make_batched
from ..afqmc_config import AFQMCConfig
from ..trial.stochastic import VAFQMCTrial
from ..trial.stochastic_internal_methods import _ensure_pool_field_batch_size
from ..utils import (
    PropagationData,
    analyze_energy_blocks,
    apply_trotter,
    build_propagation_data,
    calc_force_bias,
    calc_local_energy_batch,
    calc_slov_batch,
)
from ..utils.visualization import build_energy_visualizer
from ..walker import AFQMCState, init_walkers, stochastic_reconfiguration
from .multi_gpu import (
    PMAP_AXIS_NAME,
    distributed_stochastic_reconfiguration,
    host_gather_state,
    host_replicated_value,
    require_local_devices,
    shard_local_pytrees,
)


def _ensure_complex_walkers(walkers: Any) -> Any:
    if isinstance(walkers, tuple) and len(walkers) == 2:
        w_up, w_dn = walkers
        return (w_up.astype(jnp.complex128), w_dn.astype(jnp.complex128))
    if isinstance(walkers, jnp.ndarray):
        return walkers.astype(jnp.complex128)
    return walkers


def _log_init(logger: logging.Logger, cfg: AFQMCConfig, state: AFQMCState, start: float) -> None:
    logger.info(
        "AFQMC init: dt=%.4g walkers=%d blocks=%d ene_measurements=%d block_steps=%d meas_samples=%d pop_freq=%d",
        cfg.dt,
        cfg.n_walkers,
        cfg.n_blocks,
        max(int(getattr(cfg, "n_ene_measurements", 1)), 1),
        cfg.n_block_steps,
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


def _write_raw_block_data(path: str, rows: list[tuple[int, float, float, float, float, float]]) -> None:
    p = Path(path)
    if p.parent and str(p.parent) != "":
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write("# block e_blk e_est wsum block_weight elapsed_s\n")
        for blk, e_blk, e_est, wsum, wblk, elapsed in rows:
            f.write(
                f"{blk:d} {e_blk:.16e} {e_est:.16e} {wsum:.16e} {wblk:.16e} {elapsed:.6f}\n"
            )


def _open_raw_block_stream(path: str) -> TextIO:
    p = Path(path)
    if p.parent and str(p.parent) != "":
        p.parent.mkdir(parents=True, exist_ok=True)
    f = p.open("w", encoding="utf-8")
    f.write("# block e_blk e_est wsum block_weight elapsed_s\n")
    f.flush()
    return f


def _append_raw_block_row(
    f: TextIO, row: tuple[int, float, float, float, float, float]
) -> None:
    blk, e_blk, e_est, wsum, wblk, elapsed = row
    f.write(
        f"{blk:d} {e_blk:.16e} {e_est:.16e} {wsum:.16e} {wblk:.16e} {elapsed:.6f}\n"
    )
    f.flush()


def _log_pop_control_stats(logger: logging.Logger, state: AFQMCState, label: str) -> None:
    w = jnp.real(state.weights)
    w_min = float(jnp.min(w))
    w_max = float(jnp.max(w))
    w_mean = float(jnp.mean(w))
    w_sum = float(jnp.sum(w))
    n_zero_w = int(jnp.sum(w <= 0.0))

    logov = jnp.real(state.logov)
    finite = jnp.isfinite(logov)
    n_total = int(logov.shape[0])
    n_finite = int(jnp.sum(finite))
    n_nonfinite = n_total - n_finite
    if n_finite > 0:
        lo_min = float(jnp.min(jnp.where(finite, logov, jnp.inf)))
        lo_max = float(jnp.max(jnp.where(finite, logov, -jnp.inf)))
        lo_mean = float(jnp.sum(jnp.where(finite, logov, 0.0)) / n_finite)
    else:
        lo_min = float("nan")
        lo_max = float("nan")
        lo_mean = float("nan")

    logger.info(
        "PopCtrl %s: "
        "w_min=%.6e w_max=%.6e w_mean=%.6e w_sum=%.6e n_zero_w=%d | "
        "logov_min=%.6e logov_max=%.6e logov_mean=%.6e n_logov_nonfinite=%d",
        label,
        w_min,
        w_max,
        w_mean,
        w_sum,
        n_zero_w,
        lo_min,
        lo_max,
        lo_mean,
        n_nonfinite,
    )


def _init_pop_summary() -> dict[str, float]:
    return {
        "events": 0.0,
        "n_zero_sum": 0.0,
        "n_zero_max": 0.0,
        "wsum_min": float("inf"),
        "wsum_max": 0.0,
    }


def _update_pop_summary(summary: dict[str, float], state: AFQMCState) -> None:
    w = jnp.real(state.weights)
    n_zero = float(jnp.sum(w <= 0.0))
    wsum = float(jnp.sum(w))
    summary["events"] += 1.0
    summary["n_zero_sum"] += n_zero
    summary["n_zero_max"] = max(summary["n_zero_max"], n_zero)
    summary["wsum_min"] = min(summary["wsum_min"], wsum)
    summary["wsum_max"] = max(summary["wsum_max"], wsum)


def _merge_pop_summary(dst: dict[str, float], src: dict[str, float]) -> None:
    if src["events"] <= 0.0:
        return
    dst["events"] += src["events"]
    dst["n_zero_sum"] += src["n_zero_sum"]
    dst["n_zero_max"] = max(dst["n_zero_max"], src["n_zero_max"])
    dst["wsum_min"] = min(dst["wsum_min"], src["wsum_min"])
    dst["wsum_max"] = max(dst["wsum_max"], src["wsum_max"])


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
        + prop_data.dt * (state.pop_control_shift - prop_data.h0_prop)
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
    step_start: int,
    record_wsum: bool,
):
    ortho_freq_i = int(cfg.ortho_freq)
    step_start_i = jnp.asarray(step_start, dtype=jnp.int32)

    def body(carry, _):
        st, idx = carry
        st = _propagate_step_custom(hamil, trial, st, prop_data, cfg)
        if ortho_freq_i > 0:
            do_ortho = ((step_start_i + idx + 1) % ortho_freq_i) == 0
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
                lambda st, step0: _scan_steps_custom(
                    st,
                    self.hamil,
                    self.trial,
                    self.prop_data,
                    self.cfg,
                    n_steps_i,
                    step0,
                    False,
                )
            )
            self.cache[n_steps_i] = fn
        return fn

    def run_steps(self, state: AFQMCState, n_steps: int, label: str, step_start: int = 0) -> AFQMCState:
        n_steps_i = int(n_steps)
        if n_steps_i <= 0:
            return state

        if self.enabled:
            try:
                state, _ = self._get_scan(n_steps_i)(state, int(step_start))
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
            if self.cfg.ortho_freq > 0 and (int(step_start) + step + 1) % self.cfg.ortho_freq == 0:
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


@dataclass
class _StochasticInitContext:
    device: Any
    root_key: Any
    stage_key: Any
    walker_sampler: Any
    sampler_params: Any
    sample_state: Any
    infer_state: Any
    chains_per_walker: int


def _clone_stochastic_trial_for_runtime_init(
    trial: VAFQMCTrial,
    *,
    seed: int,
) -> VAFQMCTrial:
    clone = VAFQMCTrial(
        trial.ansatz,
        trial.params,
        hamiltonian=getattr(trial, "_hamiltonian", None),
        reference_wfn=trial.reference_wfn,
        n_samples=int(trial.n_samples),
        burn_in=int(trial.burn_in),
        sampler_name=str(trial._sampler_name),
        sampler_kwargs=dict(getattr(trial, "_sampler_kwargs", {}) or {}),
        sampling_target=str(trial.sampling_target),
        logdens_floor=float(trial.logdens_floor),
        sample_update_steps=int(trial.sample_update_steps),
        local_energy_chunk_size=int(trial.local_energy_chunk_size),
        init_walkers_from_trial=bool(trial.init_walkers_from_trial),
        init_walkers_burn_in=int(trial.init_walkers_burn_in),
        init_walkers_chains_per_walker=int(trial.init_walkers_chains_per_walker),
        init_walkers_infer_steps=int(trial.init_walkers_infer_steps),
        max_prop=trial.max_prop,
        seed=int(seed),
    )
    return clone


def _make_stochastic_init_infer_pm(trial: VAFQMCTrial):
    hamil = getattr(trial, "_hamiltonian", None)
    if hamil is None:
        return None

    apply_ansatz = jax.vmap(lambda f: trial.ansatz.apply(trial.params, f))
    eval_eloc = jax.vmap(lambda b, k: hamil.local_energy(b, k))
    eval_slov = jax.vmap(lambda b, k: calc_slov(b, k))
    floor = float(getattr(trial, "logdens_floor", -60.0))
    local_chunk = int(getattr(trial, "local_energy_chunk_size", 0))
    chunk = int(local_chunk * int(getattr(trial, "n_samples", 1))) if local_chunk > 0 else 0

    def infer_pm(fields_pair_local: Any, logsw_local: Any):
        bra_fields = tree_map(lambda x: x[:, 0], fields_pair_local)
        ket_fields = tree_map(lambda x: x[:, 1], fields_pair_local)
        bra, bra_lw = apply_ansatz(bra_fields)
        ket, ket_lw = apply_ansatz(ket_fields)
        n_local_walkers = int(tree_leaves(ket)[0].shape[0])

        if chunk <= 0 or chunk >= n_local_walkers:
            eloc = eval_eloc(bra, ket)
            sign, logabs = eval_slov(bra, ket)
        else:
            eloc_chunks = []
            sign_chunks = []
            logabs_chunks = []
            for start in range(0, n_local_walkers, chunk):
                end = min(start + chunk, n_local_walkers)
                bra_c = tree_map(lambda x: x[start:end], bra)
                ket_c = tree_map(lambda x: x[start:end], ket)
                eloc_chunks.append(eval_eloc(bra_c, ket_c))
                sign_c, logabs_c = eval_slov(bra_c, ket_c)
                sign_chunks.append(sign_c)
                logabs_chunks.append(logabs_c)
            eloc = jnp.concatenate(eloc_chunks, axis=0)
            sign = jnp.concatenate(sign_chunks, axis=0)
            logabs = jnp.concatenate(logabs_chunks, axis=0)

        logov = logabs + bra_lw + ket_lw
        dlog = jnp.real(logov - logsw_local)
        dlog = jnp.where(jnp.isfinite(dlog), dlog, floor)
        dmax = lax.pmax(jnp.max(dlog), PMAP_AXIS_NAME)
        rel_w = jnp.exp(dlog - dmax)

        global_wsum = jnp.maximum(lax.psum(jnp.sum(rel_w), PMAP_AXIS_NAME), 1.0e-12)
        global_es_num = lax.psum(jnp.sum((eloc * sign) * rel_w), PMAP_AXIS_NAME)
        global_s_num = lax.psum(jnp.sum(sign * rel_w), PMAP_AXIS_NAME)

        exp_es = global_es_num / global_wsum
        exp_s = global_s_num / global_wsum
        denom = jnp.real(exp_s)
        etot = jnp.where(jnp.abs(denom) > 1.0e-12, jnp.real(exp_es) / denom, jnp.nan)
        return jnp.real(etot), jnp.real(exp_es), jnp.real(exp_s)

    return jax.pmap(infer_pm, axis_name=PMAP_AXIS_NAME)


def _build_stochastic_init_context_local(
    trial: VAFQMCTrial,
    n_local: int,
    key: Any,
    device: Any,
) -> _StochasticInitContext:
    chains_per_walker = int(getattr(trial, "init_walkers_chains_per_walker", 0))
    if chains_per_walker <= 0:
        chains_per_walker = int(trial.n_samples)
    n_init_chains = int(n_local * chains_per_walker)

    pair_fields_shape = tree_map(
        lambda s: jnp.asarray((2, *tuple(map(int, jnp.asarray(s).reshape(-1).tolist())))),
        trial.fields_shape,
    )

    def pair_logdens(params: Any, fields_pair: Any):
        bra_fields = tree_map(lambda x: x[0], fields_pair)
        ket_fields = tree_map(lambda x: x[1], fields_pair)
        bra, bra_lw = trial.ansatz.apply(params, bra_fields)
        ket, ket_lw = trial.ansatz.apply(params, ket_fields)
        _sign, logov = calc_slov(bra, ket)
        logd = jnp.real(logov + bra_lw + ket_lw)
        return jnp.where(jnp.isfinite(logd), logd, trial.logdens_floor)

    init_sampler = choose_sampler_maker(str(trial._sampler_name))(
        pair_logdens,
        pair_fields_shape,
        **dict(getattr(trial, "_sampler_kwargs", {}) or {}),
    )
    walker_sampler = make_batched(init_sampler, n_init_chains, concat=False)
    sampler_params = trial.params

    with jax.default_device(device):
        stage_key, init_key = jax.random.split(key)
        sample_state = walker_sampler.init(init_key, sampler_params)
        if int(trial.init_walkers_burn_in) > 0:
            stage_key, burn_key = jax.random.split(stage_key)
            sample_state = walker_sampler.burn_in(
                burn_key,
                sampler_params,
                sample_state,
                int(trial.init_walkers_burn_in),
            )

    return _StochasticInitContext(
        device=device,
        root_key=key,
        stage_key=stage_key,
        walker_sampler=walker_sampler,
        sampler_params=sampler_params,
        sample_state=sample_state,
        infer_state=sample_state,
        chains_per_walker=chains_per_walker,
    )


def _sample_stochastic_infer_local(
    ctx: _StochasticInitContext,
) -> tuple[_StochasticInitContext, Any, Any]:
    with jax.default_device(ctx.device):
        stage_key, infer_key = jax.random.split(ctx.stage_key)
        infer_state, (infer_fields, infer_logsw) = ctx.walker_sampler.sample(
            infer_key,
            ctx.sampler_params,
            ctx.infer_state,
        )
    return (
        _StochasticInitContext(
            device=ctx.device,
            root_key=ctx.root_key,
            stage_key=stage_key,
            walker_sampler=ctx.walker_sampler,
            sampler_params=ctx.sampler_params,
            sample_state=ctx.sample_state,
            infer_state=infer_state,
            chains_per_walker=ctx.chains_per_walker,
        ),
        infer_fields,
        infer_logsw,
    )


def _finalize_stochastic_walkers_local(
    trial: VAFQMCTrial,
    ctx: _StochasticInitContext,
    n_local: int,
    noise: float,
):
    with jax.default_device(ctx.device):
        stage_key, sample_key = jax.random.split(ctx.stage_key)
        _, (fields, _logsw) = ctx.walker_sampler.sample(
            sample_key,
            ctx.sampler_params,
            ctx.sample_state,
        )
        fields_for_walkers = tree_map(lambda x: x[:, 1], fields)
        walkers_all, _ = jax.vmap(lambda f: trial.ansatz.apply(trial.params, f))(fields_for_walkers)
        left_fields_all = tree_map(lambda x: x[:, 0], fields)

        if int(ctx.chains_per_walker) > 1:
            stage_key, pick_key = jax.random.split(stage_key)
            pick = jax.random.randint(pick_key, (n_local,), 0, int(ctx.chains_per_walker))
            idx = jnp.arange(n_local)

            def _pick_one(x):
                x = x.reshape((n_local, int(ctx.chains_per_walker)) + x.shape[1:])
                return x[idx, pick]

            walkers = tree_map(_pick_one, walkers_all)

            def _group(x):
                return x.reshape((n_local, int(ctx.chains_per_walker)) + x.shape[1:])

            grouped_left = tree_map(_group, left_fields_all)
            pool_seed = _ensure_pool_field_batch_size(trial, grouped_left, int(trial.n_samples))
            left_selected = tree_map(_pick_one, left_fields_all)
            pool_seed = tree_map(lambda x, s: x.at[:, 0].set(s), pool_seed, left_selected)
        else:
            walkers = walkers_all
            left_selected = left_fields_all
            pool_seed = tree_map(
                lambda x: jnp.repeat(x[:, None], int(trial.n_samples), axis=1),
                left_selected,
            )

        walkers = _ensure_complex_walkers(walkers)
        if noise > 0.0:
            if isinstance(walkers, (tuple, list)) and len(walkers) == 2:
                _, k1, k2 = jax.random.split(ctx.root_key, 3)
                w_up, w_dn = walkers
                walkers = (
                    w_up + noise * jax.random.normal(k1, w_up.shape),
                    w_dn + noise * jax.random.normal(k2, w_dn.shape),
                )
            else:
                _, kn = jax.random.split(ctx.root_key, 2)
                walkers = walkers + noise * jax.random.normal(kn, walkers.shape)

        walkers, _ = orthonormalize(walkers)
    return walkers, pool_seed


def _initialize_runtime_state_local(
    trial: VAFQMCTrial,
    hamil: Hamiltonian,
    walkers: Any,
    pool_seed: Any,
    key: Any,
    device: Any,
    e_est_init: Any,
) -> tuple[AFQMCState, float]:
    trial_local = _clone_stochastic_trial_for_runtime_init(trial, seed=int(jax.device_get(key)[0]))
    trial_local._runtime_burnin_log_enabled = False
    trial_local._init_pool_fields_override = pool_seed

    with jax.default_device(device):
        key_out, trial_key = jax.random.split(key)
        trial_state = trial_local.init_runtime_state(walkers, key=trial_key, reinit=True)
        sign, logov = trial_local.calc_slov_state(walkers, trial_state)
        sign = sign.astype(jnp.complex128)
        logov = logov.astype(jnp.float64)
        weights = jnp.ones((int(tree_leaves(walkers)[0].shape[0]),), dtype=jnp.float64)
        e_sum = 0.0
        if e_est_init is None:
            e_local = jnp.sum(jnp.real(trial_local.calc_local_energy_state(hamil, walkers, trial_state)))
            e_sum = float(jax.device_get(e_local))

        state = AFQMCState(
            walkers=walkers,
            weights=weights,
            sign=sign,
            logov=logov,
            key=key_out,
            e_estimate=jnp.asarray(0.0, dtype=jnp.float64),
            pop_control_shift=jnp.asarray(0.0, dtype=jnp.float64),
            walker_fields=_initial_walker_fields_custom(trial_local, trial_state),
            trial_state=trial_state,
        )
    return state, e_sum


def _build_initial_state_custom_multi_stochastic(
    hamil: Hamiltonian,
    trial: VAFQMCTrial,
    cfg: AFQMCConfig,
    n_devices: int,
    n_local: int,
    devices: Any,
) -> tuple[PropagationData, AFQMCState, int, int]:
    logger = logging.getLogger("hafqmc.afqmc")
    prop_data = build_propagation_data(hamil, trial, cfg.dt)
    local_keys = list(jax.random.split(jax.random.PRNGKey(cfg.seed), n_devices))
    e_est_init = getattr(cfg, "e_estimate_init", None)

    chains_per_walker = int(getattr(trial, "init_walkers_chains_per_walker", 0))
    if chains_per_walker <= 0:
        chains_per_walker = int(trial.n_samples)

    if bool(getattr(trial, "init_walkers_from_trial", False)):
        logger.info(
            "Initializing walkers from trial sampler: n_walkers=%d local_walkers=%d devices=%d sampler=%s burn_in=%d",
            int(cfg.n_walkers),
            int(n_local),
            int(n_devices),
            str(getattr(trial, "_sampler_name", "unknown")),
            int(trial.init_walkers_burn_in),
        )
        logger.info(
            "Trial & walker burn-in: total_chains=%d (%d walkers x %d chains/walker)",
            int(cfg.n_walkers * chains_per_walker),
            int(cfg.n_walkers),
            int(chains_per_walker),
        )

        with ThreadPoolExecutor(max_workers=n_devices) as executor:
            init_ctx_futs = [
                executor.submit(_build_stochastic_init_context_local, trial, n_local, key_i, dev)
                for dev, key_i in zip(devices, local_keys)
            ]
            contexts = [f.result() for f in init_ctx_futs]

            infer_steps = int(getattr(trial, "init_walkers_infer_steps", 0))
            infer_pm = _make_stochastic_init_infer_pm(trial) if infer_steps > 0 else None
            if infer_steps > 0 and infer_pm is not None:
                logger.info(
                    "Starting trial energy inference for %d steps on %d devices",
                    int(infer_steps),
                    int(n_devices),
                )
                for ii in range(infer_steps):
                    step_futs = [executor.submit(_sample_stochastic_infer_local, ctx) for ctx in contexts]
                    step_out = [f.result() for f in step_futs]
                    contexts = [x[0] for x in step_out]
                    infer_fields_local = [x[1] for x in step_out]
                    infer_logsw_local = [x[2] for x in step_out]
                    infer_fields = shard_local_pytrees(infer_fields_local, devices)
                    infer_logsw = jax.device_put_sharded(infer_logsw_local, devices)
                    e_infer_rep, exp_es_rep, exp_s_rep = infer_pm(infer_fields, infer_logsw)
                    logger.info(
                        "Infer %d/%d e_vafqmc=%.12f exp_es=%.12f exp_s=%.12f",
                        ii + 1,
                        int(infer_steps),
                        float(host_replicated_value(e_infer_rep)),
                        float(host_replicated_value(exp_es_rep)),
                        float(host_replicated_value(exp_s_rep)),
                    )

            final_futs = [
                executor.submit(_finalize_stochastic_walkers_local, trial, ctx, n_local, cfg.init_noise)
                for ctx in contexts
            ]
            final_out = [f.result() for f in final_futs]
            walkers_local = [x[0] for x in final_out]
            pool_seed_local = [x[1] for x in final_out]

            if int(getattr(trial, "burn_in", 0)) > 0:
                logger.info(
                    "Burning in the trial for %d steps on %d devices.",
                    int(trial.burn_in),
                    int(n_devices),
                )

            runtime_futs = [
                executor.submit(
                    _initialize_runtime_state_local,
                    trial,
                    hamil,
                    walkers_i,
                    pool_seed_i,
                    key_i,
                    dev,
                    e_est_init,
                )
                for walkers_i, pool_seed_i, key_i, dev in zip(walkers_local, pool_seed_local, local_keys, devices)
            ]
            runtime_out = [f.result() for f in runtime_futs]
            local_states = [x[0] for x in runtime_out]
            e_sum = float(sum(x[1] for x in runtime_out))

        logger.info(
            "End of walker burn-in: total walkers=%d.",
            int(cfg.n_walkers),
        )
    else:
        e_sum = 0.0
        local_states = []
        for dev, key_i in zip(devices, local_keys):
            with jax.default_device(dev):
                walkers, key_out = init_walkers(trial, n_local, key_i, noise=cfg.init_noise)
                walkers = _ensure_complex_walkers(walkers)
                trial_state, key_out = _bind_trial_runtime_state_custom(trial, walkers, key_out)
                sign, logov = _calc_trial_overlap_custom(trial, walkers, trial_state)
                sign = sign.astype(jnp.complex128)
                logov = logov.astype(jnp.float64)
                weights = jnp.ones((n_local,), dtype=jnp.float64)
                if e_est_init is None:
                    e_local = jnp.sum(
                        _calc_trial_local_energy_custom(hamil, trial, walkers, trial_state)
                    )
                    e_sum += float(jax.device_get(e_local))
                local_states.append(
                    AFQMCState(
                        walkers=walkers,
                        weights=weights,
                        sign=sign,
                        logov=logov,
                        key=key_out,
                        e_estimate=jnp.asarray(0.0, dtype=jnp.float64),
                        pop_control_shift=jnp.asarray(0.0, dtype=jnp.float64),
                        walker_fields=_initial_walker_fields_custom(trial, trial_state),
                        trial_state=trial_state,
                    )
                )

    e_estimate = (
        jnp.asarray(float(e_est_init), dtype=jnp.float64)
        if e_est_init is not None
        else jnp.asarray(e_sum / float(cfg.n_walkers), dtype=jnp.float64)
    )
    local_states = [
        AFQMCState(
            walkers=st.walkers,
            weights=st.weights,
            sign=st.sign,
            logov=st.logov,
            key=st.key,
            e_estimate=e_estimate,
            pop_control_shift=e_estimate,
            walker_fields=st.walker_fields,
            trial_state=st.trial_state,
        )
        for st in local_states
    ]
    return prop_data, shard_local_pytrees(local_states, devices), n_devices, n_local


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

    e_est_init = getattr(cfg, "e_estimate_init", None)
    if e_est_init is None:
        e_samples = _calc_trial_local_energy_custom(hamil, trial, walkers, trial_state)
        e_estimate = jnp.sum(e_samples) / cfg.n_walkers
    else:
        e_estimate = jnp.asarray(float(e_est_init), dtype=jnp.float64)

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


def _resample_state_custom(trial: Any, state: AFQMCState) -> AFQMCState:
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


def _run_steps_with_pop_control_custom(
    state: AFQMCState,
    runner: _CustomScanRunner,
    *,
    n_steps: int,
    step_counter: int,
    pop_freq: int,
    do_resample: bool,
    trial: Any,
    label: str,
) -> tuple[AFQMCState, int, dict[str, float]]:
    steps_done = 0
    local_step = 0
    summary = _init_pop_summary()
    while steps_done < n_steps:
        if pop_freq > 0:
            to_pop = pop_freq - (step_counter % pop_freq)
            nrun = min(n_steps - steps_done, to_pop)
        else:
            nrun = n_steps - steps_done
        state = runner.run_steps(state, nrun, label, step_start=local_step)
        step_counter += nrun
        steps_done += nrun
        local_step += nrun
        if do_resample and pop_freq > 0 and step_counter % pop_freq == 0:
            _update_pop_summary(summary, state)
            state = _resample_state_custom(trial, state)
    return state, step_counter, summary


def _run_custom_equilibration(
    state: AFQMCState,
    runner: _CustomScanRunner,
    hamil: Hamiltonian,
    trial: Any,
    cfg: AFQMCConfig,
    logger: logging.Logger,
    start: float,
) -> AFQMCState:
    log_enabled = bool(getattr(cfg, "log_enabled", True))
    eq_freq = int(getattr(cfg, "log_equil_freq", 0))
    log_eq = bool(log_enabled and eq_freq > 0)
    eq_chunk = max(eq_freq, 1) if eq_freq > 0 else int(cfg.n_eq_steps)

    pop_freq = int(getattr(cfg, "pop_control_freq", 0))
    step_counter = 0
    eq_done = 0
    measure_equil_energy = bool(getattr(cfg, "measure_equil_energy", True))
    if log_eq:
        logger.info(
            "Equilibration sweeps: steps=%d, print_every=%d",
            int(cfg.n_eq_steps),
            int(eq_chunk),
        )
    while eq_done < cfg.n_eq_steps:
        nrun = min(eq_chunk, cfg.n_eq_steps - eq_done)
        if pop_freq > 0:
            state, step_counter, _ = _run_steps_with_pop_control_custom(
                state,
                runner,
                n_steps=nrun,
                step_counter=step_counter,
                pop_freq=pop_freq,
                do_resample=bool(cfg.resample),
                trial=trial,
                label="eql",
            )
        else:
            state = runner.run_steps(state, nrun, "eql")
        eq_done += nrun
        if measure_equil_energy:
            e_mix = _calc_mixed_energy_custom(hamil, trial, state)
            state = _update_e_estimate(state, e_mix)
            state = _relax_pop_control_shift(state, e_mix)

        if cfg.resample and pop_freq <= 0:
            state = _resample_state_custom(trial, state)

        if log_eq:
            if measure_equil_energy:
                logger.info(
                    "Eql %d/%d wsum=%.3e e_mix=%.12f e_est=%.12f elapsed=%.2fs",
                    eq_done,
                    cfg.n_eq_steps,
                    float(jnp.sum(state.weights)),
                    float(e_mix),
                    float(state.e_estimate),
                    time.time() - start,
                )
            else:
                logger.info(
                    "Eql %d/%d wsum=%.3e e_est=%.12f elapsed=%.2fs",
                    eq_done,
                    cfg.n_eq_steps,
                    float(jnp.sum(state.weights)),
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


def _propagate_step_custom_multi(
    hamil: Hamiltonian,
    trial: Any,
    state: AFQMCState,
    prop_data: PropagationData,
    cfg: AFQMCConfig,
    total_n_walkers: int,
    axis_name: str,
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
        + prop_data.dt * (state.pop_control_shift - prop_data.h0_prop)
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
    wsum = jnp.maximum(lax.psum(jnp.sum(weights), axis_name), 1.0e-12)
    pop_control_shift = (
        state.e_estimate
        - 0.1 * jnp.log(wsum / float(total_n_walkers)) / prop_data.dt
    )

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


def _single_step_custom_multi(
    state: AFQMCState,
    hamil: Hamiltonian,
    trial: Any,
    prop_data: PropagationData,
    cfg: AFQMCConfig,
    total_n_walkers: int,
    step_start: int,
    axis_name: str,
) -> AFQMCState:
    state = _propagate_step_custom_multi(
        hamil,
        trial,
        state,
        prop_data,
        cfg,
        total_n_walkers,
        axis_name,
    )
    if cfg.ortho_freq > 0:
        do_ortho = ((jnp.asarray(step_start, dtype=jnp.int32) + 1) % int(cfg.ortho_freq)) == 0
        state = _orthonormalize_custom(trial, state, do_ortho)
    return state


def _scan_steps_custom_multi(
    state: AFQMCState,
    hamil: Hamiltonian,
    trial: Any,
    prop_data: PropagationData,
    cfg: AFQMCConfig,
    total_n_walkers: int,
    n_steps: int,
    step_start: int,
    record_wsum: bool,
    axis_name: str,
):
    ortho_freq_i = int(cfg.ortho_freq)
    step_start_i = jnp.asarray(step_start, dtype=jnp.int32)

    def body(carry, _):
        st, idx = carry
        st = _propagate_step_custom_multi(
            hamil,
            trial,
            st,
            prop_data,
            cfg,
            total_n_walkers,
            axis_name,
        )
        if ortho_freq_i > 0:
            do_ortho = ((step_start_i + idx + 1) % ortho_freq_i) == 0
            st = _orthonormalize_custom(trial, st, do_ortho)
        wsum = lax.psum(jnp.sum(st.weights), axis_name)
        return (st, idx + 1), wsum

    (state, _), wsum_hist = lax.scan(body, (state, 0), None, length=n_steps)
    if record_wsum:
        return state, wsum_hist
    return state, jnp.zeros((0,), dtype=wsum_hist.dtype)


@dataclass
class _CustomScanRunnerMulti:
    hamil: Hamiltonian
    trial: Any
    prop_data: PropagationData
    cfg: AFQMCConfig
    logger: logging.Logger
    total_n_walkers: int
    enabled: bool = True
    cache: dict[int, Any] = field(default_factory=dict)
    step_fn: Any = None

    def _get_scan(self, n_steps: int):
        n_steps_i = int(n_steps)
        fn = self.cache.get(n_steps_i)
        if fn is None:
            fn = jax.pmap(
                lambda st, step0: _scan_steps_custom_multi(
                    st,
                    self.hamil,
                    self.trial,
                    self.prop_data,
                    self.cfg,
                    self.total_n_walkers,
                    n_steps_i,
                    step0,
                    False,
                    PMAP_AXIS_NAME,
                ),
                axis_name=PMAP_AXIS_NAME,
                in_axes=(0, None),
            )
            self.cache[n_steps_i] = fn
        return fn

    def _get_step(self):
        if self.step_fn is None:
            self.step_fn = jax.pmap(
                lambda st, step0: _single_step_custom_multi(
                    st,
                    self.hamil,
                    self.trial,
                    self.prop_data,
                    self.cfg,
                    self.total_n_walkers,
                    step0,
                    PMAP_AXIS_NAME,
                ),
                axis_name=PMAP_AXIS_NAME,
                in_axes=(0, None),
            )
        return self.step_fn

    def run_steps(self, state: AFQMCState, n_steps: int, label: str, step_start: int = 0) -> AFQMCState:
        n_steps_i = int(n_steps)
        if n_steps_i <= 0:
            return state

        if self.enabled:
            try:
                state, _ = self._get_scan(n_steps_i)(state, int(step_start))
                return state
            except Exception as exc:
                self.logger.warning(
                    "Custom-trial %s multi-gpu scan-jit failed; fallback to pmapped step loop. error=%s",
                    label,
                    str(exc),
                )
                self.enabled = False

        step_fn = self._get_step()
        for step in range(n_steps_i):
            state = step_fn(state, int(step_start) + step)
        return state


def _calc_mixed_energy_custom_multi(
    hamil: Hamiltonian,
    trial: Any,
    state: AFQMCState,
    axis_name: str,
) -> Any:
    if state.trial_state is not None and hasattr(trial, "calc_local_energy_state"):
        e_loc = jnp.real(trial.calc_local_energy_state(hamil, state.walkers, state.trial_state))
    else:
        e_loc = jnp.real(calc_local_energy_batch(hamil, trial, state.walkers))
    local_num = jnp.sum(state.weights * e_loc)
    local_den = jnp.sum(state.weights)
    global_num = lax.psum(local_num, axis_name)
    global_den = jnp.maximum(lax.psum(local_den, axis_name), 1.0e-12)
    return global_num / global_den


def _measure_block_energy_custom_multi(
    hamil: Hamiltonian,
    trial: Any,
    state: AFQMCState,
    cfg: AFQMCConfig,
    axis_name: str,
) -> tuple[AFQMCState, Any]:
    n_meas = max(int(getattr(cfg, "n_measure_samples", 1)), 1)
    if state.trial_state is not None and hasattr(trial, "measure_block_energy_state"):
        trial_state, block_energy_local, sign_cur, logov_cur = trial.measure_block_energy_state(
            hamil,
            state.walkers,
            state.weights,
            state.e_estimate,
            cfg.dt,
            n_meas,
            state.trial_state,
        )
        local_wsum = jnp.sum(state.weights)
        global_num = lax.psum(local_wsum * block_energy_local, axis_name)
        global_den = jnp.maximum(lax.psum(local_wsum, axis_name), 1.0e-12)
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
        return state, global_num / global_den

    if state.trial_state is not None and hasattr(trial, "calc_local_energy_state"):
        e_loc = jnp.real(trial.calc_local_energy_state(hamil, state.walkers, state.trial_state))
    else:
        e_loc = jnp.real(calc_local_energy_batch(hamil, trial, state.walkers))
    clip = jnp.sqrt(2.0 / cfg.dt)
    e_loc = jnp.where(jnp.abs(e_loc - state.e_estimate) > clip, state.e_estimate, e_loc)
    local_num = jnp.sum(state.weights * e_loc)
    local_den = jnp.sum(state.weights)
    global_num = lax.psum(local_num, axis_name)
    global_den = jnp.maximum(lax.psum(local_den, axis_name), 1.0e-12)
    return state, global_num / global_den


def _build_initial_state_custom_multi(
    hamil: Hamiltonian,
    trial: Any,
    cfg: AFQMCConfig,
) -> tuple[PropagationData, AFQMCState, int, int]:
    n_devices, n_local, devices = require_local_devices(cfg.n_walkers)
    if isinstance(trial, VAFQMCTrial):
        return _build_initial_state_custom_multi_stochastic(
            hamil,
            trial,
            cfg,
            n_devices,
            n_local,
            devices,
        )

    prop_data = build_propagation_data(hamil, trial, cfg.dt)
    local_keys = jax.random.split(jax.random.PRNGKey(cfg.seed), n_devices)
    e_est_init = getattr(cfg, "e_estimate_init", None)
    e_sum = 0.0
    local_states = []

    for dev, key_i in zip(devices, local_keys):
        with jax.default_device(dev):
            walkers, key_out = init_walkers(trial, n_local, key_i, noise=cfg.init_noise)
            walkers = _ensure_complex_walkers(walkers)
            trial_state, key_out = _bind_trial_runtime_state_custom(trial, walkers, key_out)
            sign, logov = _calc_trial_overlap_custom(trial, walkers, trial_state)
            sign = sign.astype(jnp.complex128)
            logov = logov.astype(jnp.float64)
            weights = jnp.ones((n_local,), dtype=jnp.float64)
            if e_est_init is None:
                e_local = jnp.sum(
                    _calc_trial_local_energy_custom(hamil, trial, walkers, trial_state)
                )
                e_sum += float(jax.device_get(e_local))
            local_states.append(
                AFQMCState(
                    walkers=walkers,
                    weights=weights,
                    sign=sign,
                    logov=logov,
                    key=key_out,
                    e_estimate=jnp.asarray(0.0, dtype=jnp.float64),
                    pop_control_shift=jnp.asarray(0.0, dtype=jnp.float64),
                    walker_fields=_initial_walker_fields_custom(trial, trial_state),
                    trial_state=trial_state,
                )
            )

    e_estimate = (
        jnp.asarray(float(e_est_init), dtype=jnp.float64)
        if e_est_init is not None
        else jnp.asarray(e_sum / float(cfg.n_walkers), dtype=jnp.float64)
    )
    local_states = [
        AFQMCState(
            walkers=st.walkers,
            weights=st.weights,
            sign=st.sign,
            logov=st.logov,
            key=st.key,
            e_estimate=e_estimate,
            pop_control_shift=e_estimate,
            walker_fields=st.walker_fields,
            trial_state=st.trial_state,
        )
        for st in local_states
    ]
    return prop_data, shard_local_pytrees(local_states, devices), n_devices, n_local


def _update_pop_summary_host(summary: dict[str, float], state: AFQMCState, pop_stats_fn: Any) -> None:
    n_zero_rep, wsum_rep = pop_stats_fn(state)
    n_zero = float(host_replicated_value(n_zero_rep))
    wsum = float(host_replicated_value(wsum_rep))
    summary["events"] += 1.0
    summary["n_zero_sum"] += n_zero
    summary["n_zero_max"] = max(summary["n_zero_max"], n_zero)
    summary["wsum_min"] = min(summary["wsum_min"], wsum)
    summary["wsum_max"] = max(summary["wsum_max"], wsum)


def _run_steps_with_pop_control_custom_multi(
    state: AFQMCState,
    runner: _CustomScanRunnerMulti,
    pop_stats_fn: Any,
    resample_fn: Any,
    *,
    n_steps: int,
    step_counter: int,
    pop_freq: int,
    do_resample: bool,
    collect_summary: bool,
    trial: Any,
    label: str,
) -> tuple[AFQMCState, int, dict[str, float]]:
    steps_done = 0
    local_step = 0
    summary = _init_pop_summary()
    while steps_done < n_steps:
        if pop_freq > 0:
            to_pop = pop_freq - (step_counter % pop_freq)
            nrun = min(n_steps - steps_done, to_pop)
        else:
            nrun = n_steps - steps_done
        state = runner.run_steps(state, nrun, label, step_start=local_step)
        step_counter += nrun
        steps_done += nrun
        local_step += nrun
        if do_resample and pop_freq > 0 and step_counter % pop_freq == 0:
            if collect_summary:
                _update_pop_summary_host(summary, state, pop_stats_fn)
            state = resample_fn(state)
    return state, step_counter, summary


def _run_custom_equilibration_multi(
    state: AFQMCState,
    runner: _CustomScanRunnerMulti,
    hamil: Hamiltonian,
    trial: Any,
    cfg: AFQMCConfig,
    logger: logging.Logger,
    start: float,
) -> AFQMCState:
    log_enabled = bool(getattr(cfg, "log_enabled", True))
    eq_freq = int(getattr(cfg, "log_equil_freq", 0))
    log_eq = bool(log_enabled and eq_freq > 0)
    eq_chunk = max(eq_freq, 1) if eq_freq > 0 else int(cfg.n_eq_steps)

    global_wsum_fn = jax.pmap(
        lambda st: lax.psum(jnp.sum(st.weights), PMAP_AXIS_NAME),
        axis_name=PMAP_AXIS_NAME,
    )
    pop_stats_fn = jax.pmap(
        lambda st: (
            lax.psum(jnp.sum(jnp.real(st.weights) <= 0.0), PMAP_AXIS_NAME),
            lax.psum(jnp.sum(st.weights), PMAP_AXIS_NAME),
        ),
        axis_name=PMAP_AXIS_NAME,
    )
    resample_fn = jax.pmap(
        lambda st: distributed_stochastic_reconfiguration(
            trial,
            st,
            PMAP_AXIS_NAME,
        ),
        axis_name=PMAP_AXIS_NAME,
    )
    mixed_energy_fn = jax.pmap(
        lambda st: _calc_mixed_energy_custom_multi(
            hamil,
            trial,
            st,
            PMAP_AXIS_NAME,
        ),
        axis_name=PMAP_AXIS_NAME,
    )
    update_e_fn = jax.pmap(
        lambda st, e_blk: _update_e_estimate(st, e_blk),
        in_axes=(0, None),
    )
    update_shift_fn = jax.pmap(
        lambda st, e_blk: _relax_pop_control_shift(st, e_blk),
        in_axes=(0, None),
    )

    pop_freq = int(getattr(cfg, "pop_control_freq", 0))
    step_counter = 0
    eq_done = 0
    measure_equil_energy = bool(getattr(cfg, "measure_equil_energy", True))
    if log_eq:
        logger.info(
            "Equilibration sweeps: steps=%d, print_every=%d",
            int(cfg.n_eq_steps),
            int(eq_chunk),
        )
    while eq_done < cfg.n_eq_steps:
        nrun = min(eq_chunk, cfg.n_eq_steps - eq_done)
        if pop_freq > 0:
            state, step_counter, _ = _run_steps_with_pop_control_custom_multi(
                state,
                runner,
                pop_stats_fn,
                resample_fn,
                n_steps=nrun,
                step_counter=step_counter,
                pop_freq=pop_freq,
                do_resample=bool(cfg.resample),
                collect_summary=False,
                trial=trial,
                label="eql",
            )
        else:
            state = runner.run_steps(state, nrun, "eql")
        eq_done += nrun
        if measure_equil_energy:
            e_mix = float(host_replicated_value(mixed_energy_fn(state)))
            state = update_e_fn(state, e_mix)
            state = update_shift_fn(state, e_mix)

        if cfg.resample and pop_freq <= 0:
            state = resample_fn(state)

        if log_eq:
            wsum = float(host_replicated_value(global_wsum_fn(state)))
            if measure_equil_energy:
                logger.info(
                    "Eql %d/%d wsum=%.3e e_mix=%.12f e_est=%.12f elapsed=%.2fs",
                    eq_done,
                    cfg.n_eq_steps,
                    wsum,
                    float(e_mix),
                    float(host_replicated_value(state.e_estimate)),
                    time.time() - start,
                )
            else:
                logger.info(
                    "Eql %d/%d wsum=%.3e e_est=%.12f elapsed=%.2fs",
                    eq_done,
                    cfg.n_eq_steps,
                    wsum,
                    float(host_replicated_value(state.e_estimate)),
                    time.time() - start,
                )
    return state


def run_afqmc_custom_multi(
    hamil: Hamiltonian,
    trial: Any,
    cfg: AFQMCConfig,
    logger: logging.Logger,
    start: float,
    *,
    return_state: bool,
    return_blocks: bool,
):
    prop_data, state, n_devices, n_local = _build_initial_state_custom_multi(hamil, trial, cfg)
    logger.info(
        "AFQMC init: dt=%.4g walkers=%d local_walkers=%d devices=%d blocks=%d "
        "ene_measurements=%d block_steps=%d meas_samples=%d pop_freq=%d",
        cfg.dt,
        cfg.n_walkers,
        n_local,
        n_devices,
        cfg.n_blocks,
        max(int(getattr(cfg, "n_ene_measurements", 1)), 1),
        cfg.n_block_steps,
        max(int(getattr(cfg, "n_measure_samples", 1)), 1),
        int(getattr(cfg, "pop_control_freq", 0)),
    )
    logger.info(
        "Init e_est=%.12f elapsed=%.2fs",
        float(host_replicated_value(state.e_estimate)),
        time.time() - start,
    )

    runner = _CustomScanRunnerMulti(hamil, trial, prop_data, cfg, logger, cfg.n_walkers)
    state = _run_custom_equilibration_multi(state, runner, hamil, trial, cfg, logger, start)
    visualizer = build_energy_visualizer(
        enabled=bool(getattr(cfg, "output_visualization_enabled", False)),
        logger=logger,
        title="AFQMC Live Energy (custom)",
        refresh_every=int(getattr(cfg, "output_visualization_refresh_every", 1)),
        show=bool(getattr(cfg, "output_visualization_show", False)),
        save_path=getattr(cfg, "output_visualization_save_path", "eblock_afqmc.png"),
    )

    block_energies: list[Any] = []
    block_weights: list[Any] = []
    raw_rows: list[tuple[int, float, float, float, float, float]] = []
    write_raw = bool(getattr(cfg, "output_write_raw", False))
    raw_path = str(getattr(cfg, "output_raw_path", "raw.dat"))
    raw_stream: TextIO | None = _open_raw_block_stream(raw_path) if write_raw else None
    n_ene_measurements = max(int(getattr(cfg, "n_ene_measurements", 1)), 1)
    pop_freq = int(getattr(cfg, "pop_control_freq", 0))
    log_pop_stats = bool(getattr(cfg, "pop_control_log_stats", False))
    log_enabled = bool(getattr(cfg, "log_enabled", True))
    block_log_freq = int(getattr(cfg, "log_block_freq", 1))
    step_counter = 0

    global_wsum_fn = jax.pmap(
        lambda st: lax.psum(jnp.sum(st.weights), PMAP_AXIS_NAME),
        axis_name=PMAP_AXIS_NAME,
    )
    pop_stats_fn = jax.pmap(
        lambda st: (
            lax.psum(jnp.sum(jnp.real(st.weights) <= 0.0), PMAP_AXIS_NAME),
            lax.psum(jnp.sum(st.weights), PMAP_AXIS_NAME),
        ),
        axis_name=PMAP_AXIS_NAME,
    )
    resample_fn = jax.pmap(
        lambda st: distributed_stochastic_reconfiguration(
            trial,
            st,
            PMAP_AXIS_NAME,
        ),
        axis_name=PMAP_AXIS_NAME,
    )
    measure_fn = jax.pmap(
        lambda st: _measure_block_energy_custom_multi(
            hamil,
            trial,
            st,
            cfg,
            PMAP_AXIS_NAME,
        ),
        axis_name=PMAP_AXIS_NAME,
    )
    update_shift_fn = jax.pmap(
        lambda st, e_blk: _relax_pop_control_shift(st, e_blk),
        in_axes=(0, None),
    )
    update_est_fn = jax.pmap(
        lambda st, e_blk: _update_e_estimate(st, e_blk),
        in_axes=(0, None),
    )

    try:
        for blk in range(cfg.n_blocks):
            e_num = 0.0
            e_den = 0.0
            pop_summary_block = _init_pop_summary()
            pop_summary_pre_block = _init_pop_summary()

            for _ in range(n_ene_measurements):
                if pop_freq > 0:
                    state, step_counter, pop_summary = _run_steps_with_pop_control_custom_multi(
                        state,
                        runner,
                        pop_stats_fn,
                        resample_fn,
                        n_steps=int(cfg.n_block_steps),
                        step_counter=step_counter,
                        pop_freq=pop_freq,
                        do_resample=bool(cfg.resample),
                        collect_summary=log_pop_stats,
                        trial=trial,
                        label="block",
                    )
                    _merge_pop_summary(pop_summary_block, pop_summary)
                else:
                    state = runner.run_steps(state, cfg.n_block_steps, "block")
                    step_counter += int(cfg.n_block_steps)
                state, e_rep = measure_fn(state)
                e_i = float(host_replicated_value(e_rep))
                w_i = float(host_replicated_value(global_wsum_fn(state)))
                e_num += w_i * e_i
                e_den += w_i
                state = update_shift_fn(state, e_i)

            if cfg.resample and pop_freq <= 0:
                if log_pop_stats:
                    _update_pop_summary_host(pop_summary_pre_block, state, pop_stats_fn)
                state = resample_fn(state)

            block_energy = e_num / max(e_den, 1.0e-12)
            block_energies.append(block_energy)
            block_weights.append(e_den)
            state = update_est_fn(state, block_energy)
            if log_pop_stats:
                _merge_pop_summary(pop_summary_block, pop_summary_pre_block)
                if pop_summary_block["events"] > 0.0:
                    ev = max(pop_summary_block["events"], 1.0)
                    logger.info(
                        "PopCtrl block=%d pre-resample: events=%d n_zero_mean=%.2f n_zero_max=%d "
                        "wsum_min=%.6e wsum_max=%.6e",
                        int(blk + 1),
                        int(pop_summary_block["events"]),
                        float(pop_summary_block["n_zero_sum"] / ev),
                        int(pop_summary_block["n_zero_max"]),
                        float(pop_summary_block["wsum_min"]),
                        float(pop_summary_block["wsum_max"]),
                    )
                else:
                    summary_now = _init_pop_summary()
                    _update_pop_summary_host(summary_now, state, pop_stats_fn)
                    logger.info(
                        "PopCtrl block=%d pre-resample: events=%d n_zero_mean=%.2f n_zero_max=%d "
                        "wsum_min=%.6e wsum_max=%.6e",
                        int(blk + 1),
                        0,
                        float(summary_now["n_zero_sum"]),
                        int(summary_now["n_zero_max"]),
                        float(summary_now["wsum_min"]),
                        float(summary_now["wsum_max"]),
                    )
            row = (
                int(blk + 1),
                float(block_energy),
                float(host_replicated_value(state.e_estimate)),
                float(host_replicated_value(global_wsum_fn(state))),
                float(e_den),
                float(time.time() - start),
            )
            raw_rows.append(row)
            if raw_stream is not None:
                _append_raw_block_row(raw_stream, row)
            visualizer.update(blk + 1, float(block_energy))

            if log_enabled and block_log_freq > 0 and (
                (blk + 1) % block_log_freq == 0 or blk + 1 == cfg.n_blocks
            ):
                logger.info(
                    "Block %d/%d e_blk=%.12f e_est=%.12f wsum=%.3e elapsed=%.2fs",
                    blk + 1,
                    cfg.n_blocks,
                    float(block_energy),
                    float(host_replicated_value(state.e_estimate)),
                    float(host_replicated_value(global_wsum_fn(state))),
                    time.time() - start,
                )
    finally:
        if raw_stream is not None:
            raw_stream.close()

    if write_raw:
        logger.info("Saved AFQMC raw block data: %s", raw_path)

    final_state = host_gather_state(state) if return_state else state
    return _finalize_outputs(
        logger,
        start,
        block_energies,
        block_weights,
        state=final_state,
        visualizer=visualizer,
        return_state=return_state,
        return_blocks=return_blocks,
    )


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
    if bool(getattr(cfg, "multi_gpu", False)):
        if int(jax.local_device_count()) > 1:
            return run_afqmc_custom_multi(
                hamil,
                trial,
                cfg,
                logger,
                start,
                return_state=return_state,
                return_blocks=return_blocks,
            )
        logger.warning(
            "cfg.propagation.multi_gpu=True but only %d local device is visible; falling back to single-device.",
            int(jax.local_device_count()),
        )

    prop_data, state = _build_initial_state_custom(hamil, trial, cfg)
    _log_init(logger, cfg, state, start)

    custom_runner = _CustomScanRunner(hamil, trial, prop_data, cfg, logger)
    state = _run_custom_equilibration(state, custom_runner, hamil, trial, cfg, logger, start)
    visualizer = build_energy_visualizer(
        enabled=bool(getattr(cfg, "output_visualization_enabled", False)),
        logger=logger,
        title="AFQMC Live Energy (custom)",
        refresh_every=int(getattr(cfg, "output_visualization_refresh_every", 1)),
        show=bool(getattr(cfg, "output_visualization_show", False)),
        save_path=getattr(cfg, "output_visualization_save_path", "eblock_afqmc.png"),
    )

    block_energies: list[Any] = []
    block_weights: list[Any] = []
    raw_rows: list[tuple[int, float, float, float, float, float]] = []
    write_raw = bool(getattr(cfg, "output_write_raw", False))
    raw_path = str(getattr(cfg, "output_raw_path", "raw.dat"))
    raw_stream: TextIO | None = _open_raw_block_stream(raw_path) if write_raw else None
    n_ene_measurements = max(int(getattr(cfg, "n_ene_measurements", 1)), 1)
    pop_freq = int(getattr(cfg, "pop_control_freq", 0))
    log_pop_stats = bool(getattr(cfg, "pop_control_log_stats", False))
    log_enabled = bool(getattr(cfg, "log_enabled", True))
    block_log_freq = int(getattr(cfg, "log_block_freq", 1))
    step_counter = 0

    try:
        for blk in range(cfg.n_blocks):
            e_num = jnp.asarray(0.0, dtype=jnp.float64)
            e_den = jnp.asarray(0.0, dtype=jnp.float64)
            pop_summary_block = _init_pop_summary()
            pop_summary_pre_block = _init_pop_summary()

            for _ in range(n_ene_measurements):
                if pop_freq > 0:
                    state, step_counter, pop_summary = _run_steps_with_pop_control_custom(
                        state,
                        custom_runner,
                        n_steps=int(cfg.n_block_steps),
                        step_counter=step_counter,
                        pop_freq=pop_freq,
                        do_resample=bool(cfg.resample),
                        trial=trial,
                        label="block",
                    )
                    _merge_pop_summary(pop_summary_block, pop_summary)
                else:
                    state = custom_runner.run_steps(state, cfg.n_block_steps, "block")
                    step_counter += int(cfg.n_block_steps)
                state, e_i = _measure_block_energy_custom(hamil, trial, state, cfg)
                w_i = jnp.maximum(jnp.sum(state.weights), 1.0e-12)
                e_num = e_num + w_i * e_i
                e_den = e_den + w_i
                state = _relax_pop_control_shift(state, e_i)

            if cfg.resample and pop_freq <= 0:
                _update_pop_summary(pop_summary_pre_block, state)
                state = _resample_state_custom(trial, state)

            block_energy = e_num / jnp.maximum(e_den, 1.0e-12)
            block_energies.append(block_energy)
            block_weights.append(e_den)
            state = _update_e_estimate(state, block_energy)
            if log_pop_stats:
                _merge_pop_summary(pop_summary_block, pop_summary_pre_block)
                if pop_summary_block["events"] > 0.0:
                    ev = max(pop_summary_block["events"], 1.0)
                    logger.info(
                        "PopCtrl block=%d pre-resample: events=%d n_zero_mean=%.2f n_zero_max=%d "
                        "wsum_min=%.6e wsum_max=%.6e",
                        int(blk + 1),
                        int(pop_summary_block["events"]),
                        float(pop_summary_block["n_zero_sum"] / ev),
                        int(pop_summary_block["n_zero_max"]),
                        float(pop_summary_block["wsum_min"]),
                        float(pop_summary_block["wsum_max"]),
                    )
                else:
                    _log_pop_control_stats(logger, state, f"block={blk + 1}")
            row = (
                int(blk + 1),
                float(block_energy),
                float(state.e_estimate),
                float(jnp.sum(state.weights)),
                float(e_den),
                float(time.time() - start),
            )
            raw_rows.append(row)
            if raw_stream is not None:
                _append_raw_block_row(raw_stream, row)
            visualizer.update(blk + 1, float(block_energy))

            if log_enabled and block_log_freq > 0 and (
                (blk + 1) % block_log_freq == 0 or blk + 1 == cfg.n_blocks
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
    finally:
        if raw_stream is not None:
            raw_stream.close()

    if write_raw:
        logger.info("Saved AFQMC raw block data: %s", raw_path)

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


__all__ = ["run_afqmc_custom", "run_afqmc_custom_multi"]
