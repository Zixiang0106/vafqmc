import logging
import time
import os
import yaml
from typing import NamedTuple

import jax
from jax import lax
from jax import numpy as jnp

from .ansatz import Ansatz
from .hamiltonian import calc_rdm, calc_slov, calc_v0, _align_rdm
from .utils import (cmult, pack_spin, unpack_spin, ravel_shape, tree_map,
                    make_expm_apply, warp_spin_expm, load_pickle, dict_to_cfg)
from .sampler import make_hamiltonian, make_batched, make_multistep
from .propagator import orthonormalize


# Container for AFQMC simulation state.
class AFQMCState(NamedTuple):
    step: int
    walkers: object
    logw: jnp.ndarray
    y_state: object
    rng: jnp.ndarray
    alive: jnp.ndarray
    et_shift: jnp.ndarray
    overlaps: object
    e_estimate: jnp.ndarray


def _extract_params(obj):
    """Extract parameter PyTree from a checkpoint payload with nested tuples."""
    if isinstance(obj, tuple):
        if len(obj) == 2 and isinstance(obj[1], tuple) and len(obj[1]) > 1:
            return obj[1][1]
        if len(obj) > 1:
            return obj[1]
    return obj


def _extract_ansatz_params(trial_params):
    """Return params tree compatible with Ansatz.apply (handle nested 'ansatz' key)."""
    if not isinstance(trial_params, dict):
        return trial_params
    params = trial_params.get("params", trial_params)
    if isinstance(params, dict) and "ansatz" in params and isinstance(params["ansatz"], dict):
        return {"params": params["ansatz"]}
    return trial_params


def _stack_walkers(wfn, nwalkers: int):
    """Tile a single walker wavefunction into a batched walker array."""
    def _stack(x):
        return jnp.broadcast_to(x, (nwalkers,) + x.shape)
    return tree_map(_stack, wfn)


def _unravel_batched(unravel_fn, flat, batch_shape):
    """Unravel flattened field vectors into the structured fields PyTree."""
    flat2 = flat.reshape((-1, flat.shape[-1]))
    fields2 = jax.vmap(unravel_fn)(flat2)
    return tree_map(lambda x: x.reshape(batch_shape + x.shape[1:]), fields2)


def _build_background(hamiltonian, bg_wfn):
    """Compute mean-field background shift and modified one-body term."""
    eri = hamiltonian.ceri if hamiltonian._eri is None else hamiltonian._eri
    hmf_raw = hamiltonian.h1e - 0.5 * calc_v0(eri)
    if bg_wfn is None:
        vbar0 = jnp.zeros((hamiltonian.ceri.shape[0],))
        return vbar0, hmf_raw, hamiltonian.enuc
    rdm_bg = calc_rdm(bg_wfn, bg_wfn)
    rdm_tot, _ = _align_rdm(rdm_bg, hamiltonian.nbasis)
    vbar0 = jnp.einsum("kpq,pq->k", hamiltonian.ceri, rdm_tot)
    hmf = hmf_raw + jnp.einsum("kpq,k->pq", hamiltonian.ceri, vbar0)
    enuc = hamiltonian.enuc - 0.5 * (vbar0**2).sum()
    return vbar0, hmf, enuc


def _build_ops(hamiltonian, dt, expm_option=("scan", 6, 1), bg_wfn=None):
    """Build Trotter propagation operators and constants."""
    vbar0, hmf, enuc = _build_background(hamiltonian, bg_wfn)
    step = jnp.sqrt(-dt + 0j)
    expm_apply = warp_spin_expm(make_expm_apply(*expm_option))
    return hmf, hamiltonian.ceri, vbar0, step, expm_apply, enuc


def _make_trial_sampler(trial, trial_params, hamiltonian, fields_shape, hmc_cfg):
    """Construct HMC sampler for trial path variables conditioned on walkers."""
    def logdens_fn(walker, fields):
        phi_t, logw_t = trial.apply(trial_params, fields)
        sign, logov = calc_slov(phi_t, walker)
        return (logov.real + logw_t).real
    sampler = make_hamiltonian(logdens_fn, fields_shape,
                               dt=hmc_cfg["dt"], length=hmc_cfg["length"])
    if hmc_cfg.get("n_chains", 1) > 1:
        sampler = make_batched(sampler, hmc_cfg["n_chains"], concat=False)
    if hmc_cfg.get("sweeps", 1) > 1:
        sampler = make_multistep(sampler, hmc_cfg["sweeps"], concat=False)
    return sampler


def _complex_sum(sign, logabs):
    """Stable weighted sum in complex space using log-sum-exp stabilization."""
    max_log = jnp.max(logabs)
    weights = jnp.exp(logabs - max_log)
    return jnp.sum(sign * weights) * jnp.exp(max_log)


def _compute_force_and_overlap(trial, trial_params, hamiltonian, vhs_raw, vbar0,
                               walker, fields):
    """Compute force-bias (vbar) and old overlap using sampled trial paths."""
    trial_apply = lambda f: trial.apply(trial_params, f)
    phi_t, logw_t = jax.vmap(trial_apply)(fields)
    sign_old, logov_old = jax.vmap(lambda phi: calc_slov(phi, walker))(phi_t)
    logabs_old = logov_old.real + logw_t
    o_old = _complex_sum(sign_old, logabs_old)

    rdm = jax.vmap(lambda phi: calc_rdm(phi, walker))(phi_t)
    rdm_tot = jax.vmap(lambda r: _align_rdm(r, hamiltonian.nbasis)[0])(rdm)
    vbar_i = jax.vmap(lambda r: jnp.einsum("kpq,pq->k", vhs_raw, r))(rdm_tot)

    max_log = jnp.max(logabs_old)
    weights = jnp.exp(logabs_old - max_log) * sign_old
    wsum = jnp.sum(weights)
    vbar = jnp.sum(weights[:, None] * vbar_i, axis=0) / wsum

    eloc_i = jax.vmap(lambda phi: hamiltonian.local_energy(phi, walker))(phi_t)
    eloc = jnp.sum(weights * eloc_i) / wsum

    return phi_t, logw_t, o_old, vbar, eloc


def _compute_overlap_from_phi(phi_t, logw_t, hamiltonian, walker):
    """Compute overlap from precomputed trial states and log weights."""
    sign_new, logov_new = jax.vmap(lambda phi: calc_slov(phi, walker))(phi_t)
    logabs_new = logov_new.real + logw_t
    return _complex_sum(sign_new, logabs_new)


def _compute_overlap_from_fields(trial, trial_params, walker, fields):
    """Compute overlap by regenerating trial states from fields."""
    trial_apply = lambda f: trial.apply(trial_params, f)
    phi_t, logw_t = jax.vmap(trial_apply)(fields)
    sign_new, logov_new = jax.vmap(lambda phi: calc_slov(phi, walker))(phi_t)
    logabs_new = logov_new.real + logw_t
    return _complex_sum(sign_new, logabs_new)


def _phase_sum_only(trial, trial_params, walker, fields):
    """Sum of overlap phases across trial paths (C++ MetroChains style)."""
    trial_apply = lambda f: trial.apply(trial_params, f)
    phi_t, logw_t = jax.vmap(trial_apply)(fields)
    sign, logov = jax.vmap(lambda phi: calc_slov(phi, walker))(phi_t)
    ov = sign * jnp.exp(logov + logw_t)
    phase = ov / (jnp.abs(ov) + 1e-12)
    return jnp.sum(phase)


def _phase_sum_and_vbar(trial, trial_params, hamiltonian, vhs_raw, walker, fields):
    """Phase-weighted force bias (align with C++ metro-chains)."""
    trial_apply = lambda f: trial.apply(trial_params, f)
    phi_t, logw_t = jax.vmap(trial_apply)(fields)
    sign, logov = jax.vmap(lambda phi: calc_slov(phi, walker))(phi_t)
    ov = sign * jnp.exp(logov + logw_t)
    phase = ov / (jnp.abs(ov) + 1e-12)
    phase_sum = jnp.sum(phase)

    rdm = jax.vmap(lambda phi: calc_rdm(phi, walker))(phi_t)
    rdm_tot = jax.vmap(lambda r: _align_rdm(r, hamiltonian.nbasis)[0])(rdm)
    vbar_i = jax.vmap(lambda r: jnp.einsum("kpq,pq->k", vhs_raw, r))(rdm_tot)
    vbar_num = jnp.sum(phase[:, None] * vbar_i, axis=0)
    vbar = vbar_num / (phase_sum + 1e-12)
    return phase_sum, vbar


def _phase_sum_and_eloc_sum(trial, trial_params, hamiltonian, walker, fields):
    """Phase sum and phase-weighted local-energy sum for measurement."""
    trial_apply = lambda f: trial.apply(trial_params, f)
    phi_t, logw_t = jax.vmap(trial_apply)(fields)
    sign, logov = jax.vmap(lambda phi: calc_slov(phi, walker))(phi_t)
    ov = sign * jnp.exp(logov + logw_t)
    phase = ov / (jnp.abs(ov) + 1e-12)
    phase_sum = jnp.sum(phase)

    eloc_i = jax.vmap(lambda phi: hamiltonian.local_energy(phi, walker))(phi_t)
    eloc_phase_sum = jnp.sum(phase * eloc_i)
    return phase_sum, eloc_phase_sum


def _ov_sum_and_eloc_sum(trial, trial_params, hamiltonian, walker, fields):
    """Overlap-weighted local-energy sum (burn-in diagnostic)."""
    trial_apply = lambda f: trial.apply(trial_params, f)
    phi_t, logw_t = jax.vmap(trial_apply)(fields)
    sign, logov = jax.vmap(lambda phi: calc_slov(phi, walker))(phi_t)
    ov = sign * jnp.exp(logov + logw_t)
    ov_sum = jnp.sum(ov)

    eloc_i = jax.vmap(lambda phi: hamiltonian.local_energy(phi, walker))(phi_t)
    eloc_ov_sum = jnp.sum(ov * eloc_i)
    return ov_sum, eloc_ov_sum


def _ov_sum_and_eloc_sum(trial, trial_params, hamiltonian, walker, fields):
    """Overlap sum and overlap-weighted local-energy sum (for burn-in diagnostics)."""
    trial_apply = lambda f: trial.apply(trial_params, f)
    phi_t, logw_t = jax.vmap(trial_apply)(fields)
    sign, logov = jax.vmap(lambda phi: calc_slov(phi, walker))(phi_t)
    ov = sign * jnp.exp(logov + logw_t)
    ov_sum = jnp.sum(ov)

    eloc_i = jax.vmap(lambda phi: hamiltonian.local_energy(phi, walker))(phi_t)
    eloc_ov_sum = jnp.sum(ov * eloc_i)
    return ov_sum, eloc_ov_sum


def _trial_overlap(trial_wfn, walker):
    sign, logov = calc_slov(trial_wfn, walker)
    return sign * jnp.exp(logov)


def _trial_force_bias(trial_wfn, walker, vhs_raw, hamiltonian):
    rdm = calc_rdm(trial_wfn, walker)
    rdm_tot, _ = _align_rdm(rdm, hamiltonian.nbasis)
    return jnp.einsum("kpq,pq->k", vhs_raw, rdm_tot)


def _trial_local_energy(trial_wfn, walker, hamiltonian):
    return hamiltonian.local_energy(trial_wfn, walker)


def _cap_logw(logw, cap):
    """Apply C++-style capWeight to log-weights for stability."""
    if cap is None or cap <= 0:
        return logw
    max_log = jnp.max(logw)
    weights = jnp.exp(logw - max_log)
    total = jnp.sum(weights)
    cap0 = total * cap
    mask = weights < cap0
    total_normal = jnp.sum(jnp.where(mask, weights, 0.0))
    n_normal = jnp.sum(mask)
    total_normal = jnp.where(n_normal > 0, total_normal * (weights.shape[0] / n_normal), total)
    cap1 = total_normal * cap
    weights = jnp.where(weights > cap1, cap1, weights)
    logw = jnp.log(jnp.clip(weights, 1e-30)) + max_log
    return logw


def _stochastic_reconfiguration(walkers, logw, key):
    """Resample walkers according to weights (SR), returning uniform logw."""
    max_log = jnp.max(logw)
    weights = jnp.exp(logw - max_log)
    total = jnp.sum(weights)
    nwalk = logw.shape[0]
    key, subkey = jax.random.split(key)
    zeta = jax.random.uniform(subkey)
    z = total * (jnp.arange(nwalk) + zeta) / nwalk
    cum = jnp.cumsum(weights)
    indices = jax.vmap(jnp.searchsorted, in_axes=(None, 0))(cum, z)
    walkers = tree_map(lambda x: x[indices], walkers)
    logw_avg = jnp.log(jnp.clip(total, 1e-30)) + max_log - jnp.log(nwalk)
    logw = jnp.full((nwalk,), logw_avg)
    return walkers, logw, key


def _stochastic_reconfiguration_weights(walkers, weights, key):
    """Resample walkers according to (real) weights, returning uniform weights."""
    nwalk = weights.shape[0]
    key, subkey = jax.random.split(key)
    zeta = jax.random.uniform(subkey)
    cumulative = jnp.cumsum(jnp.abs(weights))
    total = cumulative[-1]
    avg = total / nwalk
    z = total * (jnp.arange(nwalk) + zeta) / nwalk
    indices = jax.vmap(jnp.searchsorted, in_axes=(None, 0))(cumulative, z)
    walkers = tree_map(lambda x: x[indices], walkers)
    weights = jnp.ones(nwalk) * avg
    return walkers, weights, key


def _load_trial_cfg_from_hparams(cfg, trial_params_path, logger):
    """Try to load ansatz config from a saved hparams.yml near params."""
    cfg_path = getattr(cfg.afqmc, "trial_config_path", None)
    if cfg_path is None and trial_params_path:
        cfg_path = os.path.join(os.path.dirname(trial_params_path), "hparams.yml")
    if cfg_path is None and getattr(cfg, "log", None) is not None:
        cfg_path = getattr(cfg.log, "hpar_path", None)
    if not cfg_path or not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict) or "ansatz" not in data:
            logger.warning(f"Trial config not found in {cfg_path}, falling back to cfg.ansatz")
            return None
        return dict_to_cfg(data["ansatz"], type_safe=False, convert_dict=True)
    except Exception as exc:
        logger.warning(f"Failed to read trial config from {cfg_path}: {exc}")
        return None


def _propagate_one(hmf, vhs_raw, step, expm_apply, dt, walker, aux):
    """Apply one Trotter step to a single walker with given aux fields."""
    wfn_packed, nelec = pack_spin(walker)
    wfn = wfn_packed.astype(step.dtype)
    wfn = expm_apply(-0.5 * dt * hmf, wfn)
    vhs_sum = jnp.tensordot(aux, vhs_raw, axes=1)
    wfn = expm_apply(cmult(step, vhs_sum), wfn)
    wfn = expm_apply(-0.5 * dt * hmf, wfn)
    return unpack_spin(wfn, nelec)


def make_afqmc_step_fixed(trial_wfn, hamiltonian, hmf, vhs_raw, mf_shifts, h0_prop, step,
                          expm_apply, cfg):
    """Build one AFQMC propagation step function with fixed trial (HF baseline)."""
    dt = cfg["dt"]
    force_scale = jnp.sqrt(dt)
    force_cap = cfg["force_cap"]
    et_gamma = cfg.get("et_gamma", 0.0)
    et_update = cfg.get("et_update", False)
    stabilize_step = cfg["stabilize_step"]
    pop_cfg = cfg.get("pop_control", {})
    pop_freq = pop_cfg.get("freq", 0)
    pop_cap = pop_cfg.get("cap", 0.0)
    pop_enable = pop_cfg.get("enabled", False)

    def step_fn(state: AFQMCState):
        """AFQMC propagation by one time step."""
        key = state.rng
        key, key_aux, key_hmc = jax.random.split(key, 3)

        # Force bias and overlap from fixed trial.
        force_bias = jax.vmap(
            lambda w: _trial_force_bias(trial_wfn, w, vhs_raw, hamiltonian)
        )(state.walkers)
        ov_old = state.overlaps
        if ov_old is None:
            ov_old = jax.vmap(lambda w: _trial_overlap(trial_wfn, w))(state.walkers)

        # ad_afqmc-style importance sampling with mean-field shifts.
        field_shifts = -force_scale * (1j * force_bias - mf_shifts)
        if force_cap is not None and force_cap > 0:
            mag = jnp.abs(field_shifts)
            field_shifts = jnp.where(mag > force_cap, field_shifts * (force_cap / mag), field_shifts)

        fields = jax.random.normal(key_aux, shape=field_shifts.shape, dtype=field_shifts.real.dtype)
        shifted_fields = fields - field_shifts

        shift_term = jnp.sum(shifted_fields * mf_shifts, axis=-1)
        fb_term = jnp.sum(fields * field_shifts - 0.5 * field_shifts * field_shifts, axis=-1)

        # Propagate walkers with Trotterized propagator.
        prop_one = lambda w, a: _propagate_one(hmf, vhs_raw, step, expm_apply, dt, w, a)
        walkers_new = jax.vmap(prop_one, in_axes=(0, 0))(state.walkers, shifted_fields)

        # Overlap ratio for phaseless constraint.
        ov_new = jax.vmap(lambda w: _trial_overlap(trial_wfn, w))(walkers_new)
        ratio = ov_new / (ov_old + 1e-12)
        phase_factor = jnp.exp(-force_scale * shift_term)
        theta = jnp.angle(phase_factor * ratio)
        imp_fun = jnp.exp(
            -force_scale * shift_term + fb_term + dt * (state.et_shift + h0_prop)
        ) * ratio
        imp_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)
        imp_phaseless = jnp.array(jnp.where(jnp.isnan(imp_phaseless), 0.0, imp_phaseless))
        imp_phaseless = jnp.where(imp_phaseless < 1.0e-3, 0.0, imp_phaseless)
        imp_phaseless = jnp.where(imp_phaseless > 100.0, 0.0, imp_phaseless)

        weights_new = imp_phaseless * state.logw
        weights_new = jnp.where(weights_new > 100.0, 0.0, weights_new)
        alive = (weights_new > 0) & state.alive

        wsum = jnp.sum(weights_new)
        et_shift = state.e_estimate - 0.1 * jnp.log(jnp.clip(wsum / weights_new.shape[0], 1e-30)) / dt

        # One-step mixed estimator (used only for non-block runs).
        eloc_measure = jax.vmap(
            lambda w: _trial_local_energy(trial_wfn, w, hamiltonian)
        )(walkers_new).real
        clip_ref = state.e_estimate
        clip_thr = jnp.sqrt(2.0 / dt)
        eloc_measure = jnp.where(jnp.abs(eloc_measure - clip_ref) > clip_thr, clip_ref, eloc_measure)
        block_weight = jnp.sum(weights_new)
        block_energy = jnp.sum(weights_new * eloc_measure) / jnp.clip(block_weight, 1e-30)

        new_state = AFQMCState(
            state.step + 1,
            walkers_new,
            weights_new,
            state.y_state,
            key,
            alive,
            et_shift,
            ov_new,
            state.e_estimate,
        )
        stats = {
            "eloc": eloc_measure,
            "block_energy": block_energy,
            "block_weight": block_weight,
            "phase": theta,
            "alive": alive,
            "et": et_shift,
        }
        return new_state, stats

    return step_fn


def make_afqmc_step_hmc(trial, trial_params, hamiltonian, hmf, vhs_raw, vbar0, mf_shifts,
                        step, expm_apply, sampler, unravel_fn, cfg):
    """Build one AFQMC propagation step function using HMC trial paths."""
    n_chains = cfg["hmc"]["n_chains"]
    dt = cfg["dt"]
    force_scale = jnp.sqrt(dt)
    force_cap = cfg["force_cap"]
    et_gamma = cfg.get("et_gamma", 0.0)
    et_update = cfg.get("et_update", False)
    stabilize_step = cfg["stabilize_step"]
    phase_metro = cfg.get("phase_metro", False)
    pop_cfg = cfg.get("pop_control", {})
    pop_freq = pop_cfg.get("freq", 0)
    pop_cap = pop_cfg.get("cap", 0.0)
    pop_enable = pop_cfg.get("enabled", False)

    def step_fn(state: AFQMCState):
        key = state.rng
        key, key_aux, key_hmc = jax.random.split(key, 3)

        flat_fields = state.y_state[0]
        trial_fields = _unravel_batched(unravel_fn, flat_fields, (state.logw.shape[0], n_chains))

        per_walker = lambda w, f: _phase_sum_and_vbar(
            trial, trial_params, hamiltonian, vhs_raw, w, f
        )
        phase_sum_old, vbar = jax.vmap(per_walker, in_axes=(0, 0))(state.walkers, trial_fields)

        field_shifts = -force_scale * (1j * vbar - mf_shifts)
        if force_cap is not None and force_cap > 0:
            mag = jnp.abs(field_shifts)
            field_shifts = jnp.where(mag > force_cap,
                                     field_shifts * (force_cap / mag),
                                     field_shifts)

        noise = jax.random.normal(
            key_aux, shape=field_shifts.shape, dtype=field_shifts.real.dtype
        )
        shifted_fields = noise - field_shifts
        shift_term = jnp.sum(shifted_fields * mf_shifts, axis=-1)
        fb_term = jnp.sum(noise * field_shifts - 0.5 * field_shifts * field_shifts, axis=-1)

        prop_one = lambda w, a: _propagate_one(hmf, vhs_raw, step, expm_apply, dt, w, a)
        walkers_new = jax.vmap(prop_one, in_axes=(0, 0))(state.walkers, shifted_fields)

        phase_sum_new = jax.vmap(
            lambda w, f: _phase_sum_only(trial, trial_params, w, f),
            in_axes=(0, 0),
        )(walkers_new, trial_fields)
        ratio = phase_sum_new / (phase_sum_old + 1e-12)
        phase_af = jnp.angle(ratio)
        log_abs_ratio = jnp.log(jnp.abs(ratio) + 1e-12)

        logw_base = state.logw + fb_term - force_scale * shift_term + dt * state.et_shift + log_abs_ratio

        keys = jax.random.split(key_hmc, state.logw.shape[0])
        sample_one = lambda k, w, s: sampler.sample(k, w, s)
        y_state_new, _ = jax.vmap(sample_one, in_axes=(0, 0, 0))(keys, walkers_new, state.y_state)

        flat_fields_new = y_state_new[0]
        fields_new = _unravel_batched(unravel_fn, flat_fields_new, (state.logw.shape[0], n_chains))
        if phase_metro:
            phase_sum_after = jax.vmap(
                lambda w, f: _phase_sum_only(trial, trial_params, w, f),
                in_axes=(0, 0),
            )(walkers_new, fields_new)
            phase_metro_val = jnp.angle(phase_sum_after / (phase_sum_new + 1e-12))
            phase_whole = phase_af + phase_metro_val
        else:
            phase_whole = phase_af

        cos_phase = jnp.cos(phase_whole)
        alive = (cos_phase > 0) & state.alive
        logw_new = logw_base + jnp.log(jnp.clip(cos_phase, 1e-12))
        logw_new = jnp.where(alive, logw_new, -1e30)

        # Second phaseless projection: remove residual phase in logw (C++ post-step).
        phase2 = jnp.imag(logw_new)
        cos2 = jnp.cos(phase2)
        alive = (cos2 > 0) & alive
        logw_new = jnp.real(logw_new) + jnp.log(jnp.clip(cos2, 1e-12))
        logw_new = jnp.where(alive, logw_new, -1e30)
        # Normalize log-weights for numerical stability (constant shift).
        logw_new = logw_new - jnp.max(logw_new)

        if pop_enable and pop_freq and pop_freq > 0:
            do_pop = ((state.step + 1) % pop_freq) == 0

            def _apply_pop(args):
                walkers_i, logw_i, key_i, alive_i = args
                logw_i = _cap_logw(logw_i, pop_cap)
                walkers_i, logw_i, key_i = _stochastic_reconfiguration(walkers_i, logw_i, key_i)
                alive_i = jnp.ones_like(alive_i, dtype=bool)
                return walkers_i, logw_i, key_i, alive_i

            walkers_new, logw_new, key, alive = lax.cond(
                do_pop,
                _apply_pop,
                lambda x: x,
                (walkers_new, logw_new, key, alive),
            )
            # C++ popControl resets weights to unity after resampling.
            logw_new = jnp.zeros_like(logw_new)

        if stabilize_step and stabilize_step > 0:
            do_stab = ((state.step + 1) % stabilize_step) == 0

            def _apply_stab(args):
                walkers_i, logw_i = args
                walkers_i, logd = jax.vmap(orthonormalize)(walkers_i)
                logw_i = logw_i + logd
                return walkers_i, logw_i

            walkers_new, logw_new = lax.cond(
                do_stab,
                _apply_stab,
                lambda x: x,
                (walkers_new, logw_new),
            )
            # Re-normalize after QR stabilization to avoid exponential drift.
            logw_new = logw_new - jnp.max(logw_new)

        phase_sum_meas, eloc_phase_sum = jax.vmap(
            lambda w, f: _phase_sum_and_eloc_sum(trial, trial_params, hamiltonian, w, f),
            in_axes=(0, 0),
        )(walkers_new, fields_new)
        # Remove the global phase of each MetroChains sum (C++ phase_rotate).
        phase_rotate = jnp.angle(phase_sum_meas)
        phase_sum_meas = phase_sum_meas * jnp.exp(-1.0j * phase_rotate)
        eloc_phase_sum = eloc_phase_sum * jnp.exp(-1.0j * phase_rotate)

        den = jnp.exp(logw_new) * phase_sum_meas
        hsum = jnp.exp(logw_new) * eloc_phase_sum
        block_weight = jnp.sum(den)
        block_energy = (jnp.sum(hsum) / jnp.where(jnp.abs(block_weight) > 1e-30,
                                                  block_weight, (1e-30 + 0j))).real

        et_shift = state.et_shift
        if et_update and et_gamma > 0:
            pop_log = jnp.log(jnp.clip(block_weight.real, 1e-30)) - jnp.log(phase_sum_meas.shape[0])
            et_shift = block_energy - et_gamma * pop_log / dt

        new_state = AFQMCState(
            state.step + 1,
            walkers_new,
            logw_new,
            y_state_new,
            key,
            alive,
            et_shift,
            state.overlaps,
            state.e_estimate,
        )
        stats = {
            "eloc": (eloc_phase_sum / (phase_sum_meas + 1e-12)).real,
            "block_energy": block_energy,
            "block_weight": block_weight.real,
            "phase": phase_whole,
            "alive": alive,
            "et": et_shift,
        }
        return new_state, stats

    return step_fn


def run(cfg):
    """Run a full AFQMC projection using VAFQMC trial parameters."""
    logging.basicConfig(force=True, format="# [%(asctime)s] %(levelname)s: %(message)s")
    logger = logging.getLogger("afqmc")
    logger.setLevel(getattr(logging, cfg.log.level.upper()))
    start_time = time.time()

    if cfg.restart.hamiltonian is None:
        raise ValueError("AFQMC requires cfg.restart.hamiltonian to be set.")
    hamil_data = load_pickle(cfg.restart.hamiltonian)
    from .hamiltonian import Hamiltonian_sym
    hamiltonian = Hamiltonian_sym(*hamil_data)

    trial_params_path = (cfg.afqmc.trial_params
                         if "trial_params" in cfg.afqmc and cfg.afqmc.trial_params
                         else cfg.restart.params)
    if trial_params_path is None:
        raise ValueError("AFQMC requires cfg.afqmc.trial_params or cfg.restart.params.")
    trial_cfg = cfg.afqmc.trial if ("trial" in cfg.afqmc and cfg.afqmc.trial) else None
    if trial_cfg is None:
        trial_cfg = _load_trial_cfg_from_hparams(cfg, trial_params_path, logger)
    if trial_cfg is None:
        trial_cfg = cfg.ansatz
    trial = Ansatz.create(hamiltonian, **trial_cfg)
    params_raw = load_pickle(trial_params_path)
    trial_params = _extract_ansatz_params(_extract_params(params_raw))
    trial_mode = getattr(cfg.afqmc, "trial_mode", "hf").lower()
    trial_wfn = hamiltonian.wfn0

    bg_flag = cfg.afqmc.force_background
    bg_wfn = hamiltonian.wfn0 if bg_flag and str(bg_flag).lower() == "hf" else None
    hmf, vhs_raw, vbar0, step, expm_apply, enuc = _build_ops(
        hamiltonian, cfg.afqmc.dt, cfg.afqmc.expm_option, bg_wfn)
    rdm_trial = calc_rdm(trial_wfn, trial_wfn)
    rdm_tot, _ = _align_rdm(rdm_trial, hamiltonian.nbasis)
    mf_shifts = 1j * jnp.einsum("kpq,pq->k", vhs_raw, rdm_tot)
    h0_prop = -enuc - 0.5 * jnp.sum(mf_shifts**2)

    fields_shape = trial.fields_shape()
    fsize, unravel_fn = ravel_shape(fields_shape)

    sampler = None
    y_state = None

    n_walkers = cfg.afqmc.walkers
    walkers = _stack_walkers(hamiltonian.wfn0, n_walkers)
    walkers = tree_map(lambda x: x.astype(step.dtype), walkers)
    logw = jnp.zeros((n_walkers,))
    alive = jnp.ones((n_walkers,), dtype=bool)

    key = jax.random.PRNGKey(cfg.seed or 0)
    if trial_mode == "hmc":
        sampler = _make_trial_sampler(trial, trial_params, hamiltonian, fields_shape, cfg.afqmc.hmc)
        key, key_init = jax.random.split(key)
        keys = jax.random.split(key_init, n_walkers)
        init_one = lambda k, w: sampler.init(k, w)
        y_state = jax.vmap(init_one, in_axes=(0, 0))(keys, walkers)

        burn_in = cfg.afqmc.hmc.get("burn_in", 0)
        if burn_in > 0:
            keys = jax.random.split(key, n_walkers)
            burn_one = lambda k, w, s: sampler.burn_in(k, w, s, burn_in)
            y_state = jax.vmap(burn_one, in_axes=(0, 0, 0))(keys, walkers, y_state)
            if cfg.afqmc.get("log_burnin_energy", True):
                flat_fields = y_state[0]
                fields = _unravel_batched(
                    unravel_fn, flat_fields, (n_walkers, cfg.afqmc.hmc.n_chains)
                )
                per_walker = lambda w, f: _ov_sum_and_eloc_sum(
                    trial, trial_params, hamiltonian, w, f
                )
                ov_sum, eloc_ov_sum = jax.vmap(per_walker, in_axes=(0, 0))(walkers, fields)
                eloc = eloc_ov_sum / (ov_sum + 1e-12)
                logger.info(f"burn-in E={float(jnp.mean(eloc).real):.6f}")

    if trial_mode == "hf":
        overlaps = jax.vmap(lambda w: _trial_overlap(trial_wfn, w))(walkers)
        e_samples = jax.vmap(lambda w: _trial_local_energy(trial_wfn, w, hamiltonian))(walkers).real
        e_estimate = jnp.array(jnp.sum(e_samples) / n_walkers)
        logw = jnp.ones((n_walkers,))
        et_shift = e_estimate
    else:
        overlaps = None
        e_estimate = jnp.array(cfg.afqmc.ET)
        et_shift = jnp.array(cfg.afqmc.ET)

    state = AFQMCState(0, walkers, logw, y_state, key, alive, et_shift, overlaps, e_estimate)
    icf = getattr(cfg.afqmc, "icf", None)
    icf_enabled = icf is not None and icf.enabled and (trial_mode == "hmc")

    if trial_mode == "hmc":
        step_fn = make_afqmc_step_hmc(
            trial, trial_params, hamiltonian, hmf, vhs_raw, vbar0, mf_shifts, step,
            expm_apply, sampler, unravel_fn, {
                "dt": cfg.afqmc.dt,
                "force_cap": cfg.afqmc.force_cap,
                "stabilize_step": cfg.afqmc.stabilize_step,
                "hmc": {"n_chains": cfg.afqmc.hmc.n_chains},
                "phase_metro": cfg.afqmc.phase_metro,
                "pop_control": cfg.afqmc.pop_control,
                # Align with C++ MCMC-trial behavior: no per-step ET update.
                "et_update": False,
                "et_gamma": cfg.afqmc.et_update.gamma,
            },
        )
    else:
        step_fn = make_afqmc_step_fixed(
            trial_wfn, hamiltonian, hmf, vhs_raw, mf_shifts, h0_prop, step,
            expm_apply, {
                "dt": cfg.afqmc.dt,
                "force_cap": cfg.afqmc.force_cap,
                "stabilize_step": cfg.afqmc.stabilize_step,
                "pop_control": cfg.afqmc.pop_control,
                "et_update": cfg.afqmc.et_update.enabled and (not icf_enabled),
                "et_gamma": cfg.afqmc.et_update.gamma,
            },
        )

    def scan_body(carry, _):
        new_state, stats = step_fn(carry)
        return new_state, stats

    blocking = getattr(cfg.afqmc, "blocking", None)
    blocking_enabled = (blocking is not None and blocking.enabled) or icf_enabled
    if blocking_enabled:
        writes = int(blocking.writes)
        measures = int(blocking.measures)
        skip_steps = int(blocking.skip_steps)
        equil_writes = int(getattr(blocking, "equil_writes", 0))

        if trial_mode == "hmc":
            meas_thermal = int(cfg.afqmc.hmc.get("measure_thermal_sweeps", 0))
            meas_sweeps = int(cfg.afqmc.hmc.get("measure_sweeps", 1))

            def _hmc_measure_from_state(walkers, logw, y_state):
                flat_fields = y_state[0]
                fields = _unravel_batched(
                    unravel_fn, flat_fields, (logw.shape[0], cfg.afqmc.hmc.n_chains)
                )
                per_walker = lambda w, f: _phase_sum_and_eloc_sum(
                    trial, trial_params, hamiltonian, w, f
                )
                phase_sum, eloc_phase_sum = jax.vmap(
                    per_walker, in_axes=(0, 0)
                )(walkers, fields)
                phase_rotate = jnp.angle(phase_sum)
                phase_sum = phase_sum * jnp.exp(-1.0j * phase_rotate)
                eloc_phase_sum = eloc_phase_sum * jnp.exp(-1.0j * phase_rotate)
                den = jnp.exp(logw) * phase_sum
                den_sum = jnp.sum(den)
                h_sum = jnp.sum(jnp.exp(logw) * eloc_phase_sum)
                return den_sum, h_sum

            def measure_fn(carry):
                key = carry.rng
                y_state = carry.y_state
                if meas_thermal > 0:
                    key, sub = jax.random.split(key)
                    keys = jax.random.split(sub, carry.logw.shape[0])
                    burn_one = lambda k, w, s: sampler.burn_in(k, w, s, meas_thermal)
                    y_state = jax.vmap(burn_one, in_axes=(0, 0, 0))(keys, carry.walkers, y_state)
                carry = carry._replace(rng=key, y_state=y_state)

                def _one_sweep(c, _):
                    key = c.rng
                    key, sub = jax.random.split(key)
                    keys = jax.random.split(sub, c.logw.shape[0])
                    sample_one = lambda k, w, s: sampler.sample(k, w, s)
                    y_state_new, _ = jax.vmap(sample_one, in_axes=(0, 0, 0))(
                        keys, c.walkers, c.y_state
                    )
                    den_sum, h_sum = _hmc_measure_from_state(c.walkers, c.logw, y_state_new)
                    c = c._replace(rng=key, y_state=y_state_new)
                    return c, (den_sum, h_sum)

                if meas_sweeps > 0:
                    carry, (den_seq, h_seq) = lax.scan(_one_sweep, carry, xs=None, length=meas_sweeps)
                    den_sum = jnp.sum(den_seq)
                    h_sum = jnp.sum(h_seq)
                else:
                    den_sum, h_sum = _hmc_measure_from_state(carry.walkers, carry.logw, carry.y_state)

                return carry, {
                    "den_sum": den_sum,
                    "h_sum": h_sum,
                    "alive": carry.alive,
                    "et": carry.et_shift,
                }
            def measure_chunk(carry, _):
                carry, stats_now = measure_fn(carry)
                if skip_steps > 0:
                    carry, _ = lax.scan(scan_body, carry, xs=None, length=skip_steps)
                return carry, stats_now

            def block_body(carry, _):
                carry, stats_seq = lax.scan(measure_chunk, carry, xs=None, length=measures)
                den_sum = jnp.sum(stats_seq["den_sum"])
                h_sum = jnp.sum(stats_seq["h_sum"])
                denom = jnp.where(jnp.abs(den_sum) > 1e-30, den_sum, (1e-30 + 0j))
                block_energy = (h_sum / denom).real
                block_weight = den_sum.real
                stats_block = {
                    "block_energy": block_energy,
                    "block_weight": block_weight,
                    "alive": stats_seq["alive"][-1],
                    "et": stats_seq["et"][-1],
                }
                return carry, stats_block
        else:
            n_prop_steps = int(skip_steps)
            n_ene_blocks = int(measures)
            n_sr_blocks = int(getattr(blocking, "sr_blocks", 1))
            sqrt_dt = jnp.sqrt(cfg.afqmc.dt)

            def _fixed_step_with_fields(carry, fields):
                force_bias = jax.vmap(
                    lambda w: _trial_force_bias(trial_wfn, w, vhs_raw, hamiltonian)
                )(carry.walkers)
                ov_old = carry.overlaps
                if ov_old is None:
                    ov_old = jax.vmap(lambda w: _trial_overlap(trial_wfn, w))(carry.walkers)

                field_shifts = -sqrt_dt * (1.0j * force_bias - mf_shifts)
                if cfg.afqmc.force_cap is not None and cfg.afqmc.force_cap > 0:
                    mag = jnp.abs(field_shifts)
                    field_shifts = jnp.where(mag > cfg.afqmc.force_cap,
                                             field_shifts * (cfg.afqmc.force_cap / mag),
                                             field_shifts)

                shifted_fields = fields - field_shifts
                shift_term = jnp.sum(shifted_fields * mf_shifts, axis=-1)
                fb_term = jnp.sum(fields * field_shifts - field_shifts * field_shifts / 2.0, axis=-1)

                prop_one = lambda w, a: _propagate_one(hmf, vhs_raw, step, expm_apply, cfg.afqmc.dt, w, a)
                walkers_new = jax.vmap(prop_one, in_axes=(0, 0))(carry.walkers, shifted_fields)

                ov_new = jax.vmap(lambda w: _trial_overlap(trial_wfn, w))(walkers_new)
                ratio = ov_new / (ov_old + 1e-12)
                phase_factor = jnp.exp(-sqrt_dt * shift_term)
                theta = jnp.angle(phase_factor * ratio)
                imp_fun = (
                    jnp.exp(-sqrt_dt * shift_term + fb_term + cfg.afqmc.dt * (carry.et_shift + h0_prop))
                    * ratio
                )
                imp_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)
                imp_phaseless = jnp.array(jnp.where(jnp.isnan(imp_phaseless), 0.0, imp_phaseless))
                imp_phaseless = jnp.where(imp_phaseless < 1.0e-3, 0.0, imp_phaseless)
                imp_phaseless = jnp.where(imp_phaseless > 100.0, 0.0, imp_phaseless)

                weights_new = imp_phaseless * carry.logw
                weights_new = jnp.where(weights_new > 100.0, 0.0, weights_new)
                alive = weights_new > 0

                wsum = jnp.sum(weights_new)
                et_shift = carry.e_estimate - 0.1 * jnp.log(jnp.clip(wsum / weights_new.shape[0], 1e-30)) / cfg.afqmc.dt

                new_state = AFQMCState(
                    carry.step + 1,
                    walkers_new,
                    weights_new,
                    carry.y_state,
                    carry.rng,
                    alive,
                    et_shift,
                    ov_new,
                    carry.e_estimate,
                )
                return new_state, None

            def _block_scan(carry, _):
                key, subkey = jax.random.split(carry.rng)
                fields = jax.random.normal(
                    subkey,
                    shape=(n_prop_steps, n_walkers, vhs_raw.shape[0]),
                    dtype=mf_shifts.real.dtype,
                )
                carry = carry._replace(rng=key)
                if n_prop_steps > 0:
                    carry, _ = lax.scan(_fixed_step_with_fields, carry, fields)

                walkers_ortho, _ = jax.vmap(orthonormalize)(carry.walkers)
                carry = carry._replace(walkers=walkers_ortho)
                overlaps = jax.vmap(lambda w: _trial_overlap(trial_wfn, w))(carry.walkers)
                carry = carry._replace(overlaps=overlaps)

                energy_samples = jax.vmap(
                    lambda w: _trial_local_energy(trial_wfn, w, hamiltonian)
                )(carry.walkers).real
                energy_samples = jnp.where(
                    jnp.abs(energy_samples - carry.e_estimate) > jnp.sqrt(2.0 / cfg.afqmc.dt),
                    carry.e_estimate,
                    energy_samples,
                )
                block_weight = jnp.sum(carry.logw)
                block_energy = jnp.sum(energy_samples * carry.logw) / jnp.clip(block_weight, 1e-30)

                et_shift = 0.9 * carry.et_shift + 0.1 * block_energy
                carry = carry._replace(et_shift=et_shift)
                return carry, (block_energy, block_weight)

            def _sr_block_scan(carry, _):
                carry, (energy_seq, weight_seq) = lax.scan(
                    _block_scan, carry, xs=None, length=n_ene_blocks
                )
                walkers_new, weights_new, key = _stochastic_reconfiguration_weights(
                    carry.walkers, carry.logw, carry.rng
                )
                overlaps = jax.vmap(lambda w: _trial_overlap(trial_wfn, w))(walkers_new)
                carry = carry._replace(
                    walkers=walkers_new,
                    logw=weights_new,
                    rng=key,
                    overlaps=overlaps,
                    alive=weights_new > 0,
                )
                return carry, (energy_seq, weight_seq)

            def block_body(carry, _):
                overlaps = jax.vmap(lambda w: _trial_overlap(trial_wfn, w))(carry.walkers)
                carry = carry._replace(et_shift=carry.e_estimate, overlaps=overlaps)

                carry, (energy_seq, weight_seq) = lax.scan(
                    _sr_block_scan, carry, xs=None, length=n_sr_blocks
                )
                block_weight = jnp.sum(weight_seq)
                block_energy = jnp.sum(energy_seq * weight_seq) / jnp.clip(block_weight, 1e-30)

                walkers_ortho, _ = jax.vmap(orthonormalize)(carry.walkers)
                carry = carry._replace(walkers=walkers_ortho)
                walkers_new, weights_new, key = _stochastic_reconfiguration_weights(
                    carry.walkers, carry.logw, carry.rng
                )
                overlaps = jax.vmap(lambda w: _trial_overlap(trial_wfn, w))(walkers_new)
                carry = carry._replace(
                    walkers=walkers_new,
                    logw=weights_new,
                    rng=key,
                    overlaps=overlaps,
                    alive=weights_new > 0,
                )

                e_estimate = 0.9 * carry.e_estimate + 0.1 * block_energy
                carry = carry._replace(e_estimate=e_estimate)

                stats_block = {
                    "block_energy": block_energy,
                    "block_weight": block_weight,
                    "alive": carry.alive,
                    "et": carry.et_shift,
                }
                return carry, stats_block

        if icf_enabled and icf.thermal_steps > 0:
            step_fn_jit = jax.jit(step_fn)
            measure_fn_jit = jax.jit(measure_fn)
            et_acc_energy = 0.0
            et_acc_weight = 0.0
            for i in range(int(icf.thermal_steps)):
                state, _ = step_fn_jit(state)
                if icf.et_adjust_step > 0 and i < icf.et_adjust_max and (i % icf.et_adjust_step == 0):
                    state, stats_now = measure_fn_jit(state)
                    den = stats_now["den_sum"]
                    hsum = stats_now["h_sum"]
                    denom = jnp.where(jnp.abs(den) > 1e-30, den, (1e-30 + 0j))
                    new_et = float((hsum / denom).real)
                    state = state._replace(et_shift=jnp.array(new_et))
                elif (icf.et_bg_estimate_step > 0 and i >= icf.et_adjust_max
                      and i < icf.et_bg_estimate_max):
                    if ((i + 1 - icf.et_adjust_max) % icf.et_bg_estimate_step) == 0:
                        state, stats_now = measure_fn_jit(state)
                        den = float(stats_now["den_sum"].real)
                        hsum = float(stats_now["h_sum"].real)
                        et_acc_energy += hsum
                        et_acc_weight += den
                    if i == (icf.et_bg_estimate_max - 1) and et_acc_weight > 0.0:
                        new_et = et_acc_energy / et_acc_weight
                        state = state._replace(et_shift=jnp.array(new_et))
                        et_acc_energy = 0.0
                        et_acc_weight = 0.0

        if cfg.log.stat_freq > 0:
            stat_freq = int(cfg.log.stat_freq)
            block_step = jax.jit(lambda st: block_body(st, None))
            if equil_writes > 0:
                equil_freq = max(equil_writes // 5, 1)
                for ii in range(equil_writes):
                    state, stats_block = block_step(state)
                    if ii % equil_freq == 0:
                        block_energy = stats_block["block_energy"]
                        block_weight = stats_block["block_weight"]
                        alive_count = jnp.sum(stats_block["alive"])
                        et_now = stats_block["et"]
                        elapsed = time.time() - start_time
                        logger.info(
                            "eql=%d E=% .6f W=% .6f alive=%d ET=% .6f t=%.2fs",
                            ii,
                            float(block_energy),
                            float(block_weight),
                            int(alive_count),
                            float(et_now),
                            elapsed,
                        )
            for ii in range(writes):
                state, stats_block = block_step(state)
                if ii % stat_freq == 0:
                    block_energy = stats_block["block_energy"]
                    block_weight = stats_block["block_weight"]
                    alive_count = jnp.sum(stats_block["alive"])
                    et_now = stats_block["et"]
                    elapsed = time.time() - start_time
                    logger.info(
                        "block=%d E=% .6f W=% .6f alive=%d ET=% .6f t=%.2fs",
                        ii,
                        float(block_energy),
                        float(block_weight),
                        int(alive_count),
                        float(et_now),
                        elapsed,
                    )
        else:
            run_scan = jax.jit(lambda st: lax.scan(block_body, st, xs=None, length=writes))
            state, _ = run_scan(state)
    else:
        steps = int(cfg.afqmc.steps)
        run_scan = jax.jit(lambda st: lax.scan(scan_body, st, xs=None, length=steps))
        state, stats = run_scan(state)

        if cfg.log.stat_freq > 0:
            stat_freq = int(cfg.log.stat_freq)
            for ii in range(0, steps, stat_freq):
                block_energy = stats["block_energy"][ii]
                block_weight = stats["block_weight"][ii]
                alive_count = jnp.sum(stats["alive"][ii])
                et_now = stats["et"][ii]
                logger.info(
                    "step=%d E=% .6f W=% .6f alive=%d ET=% .6f",
                    ii,
                    float(block_energy),
                    float(block_weight),
                    int(alive_count),
                    float(et_now),
                )

    return state
