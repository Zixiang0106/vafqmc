import logging
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


def _extract_params(obj):
    """Extract parameter PyTree from a checkpoint payload with nested tuples."""
    if isinstance(obj, tuple):
        if len(obj) == 2 and isinstance(obj[1], tuple) and len(obj[1]) > 1:
            return obj[1][1]
        if len(obj) > 1:
            return obj[1]
    return obj


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
    wfn = expm_apply(-0.5 * dt * hmf, wfn_packed)
    vhs_sum = jnp.tensordot(aux, vhs_raw, axes=1)
    wfn = expm_apply(cmult(step, vhs_sum), wfn)
    wfn = expm_apply(-0.5 * dt * hmf, wfn)
    return unpack_spin(wfn, nelec)


def make_afqmc_step(trial, trial_params, hamiltonian, hmf, vhs_raw, vbar0, step,
                    expm_apply, sampler, unravel_fn, cfg):
    """Build one AFQMC propagation step function."""
    n_chains = cfg["hmc"]["n_chains"]
    dt = cfg["dt"]
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
        """AFQMC propagation by one time step."""
        key = state.rng
        key, key_aux, key_hmc = jax.random.split(key, 3)

        flat_fields = state.y_state[0]
        fields = _unravel_batched(unravel_fn, flat_fields, (state.logw.shape[0], n_chains))

        # Force bias and overlap from current trial paths.
        per_walker = lambda w, f: _compute_force_and_overlap(
            trial, trial_params, hamiltonian, vhs_raw, vbar0, w, f)
        phi_t, logw_t, o_old, vbar, eloc = jax.vmap(per_walker, in_axes=(0, 0))(state.walkers, fields)

        force = step * (vbar - vbar0)
        if force_cap is not None and force_cap > 0:
            mag = jnp.abs(force)
            force = jnp.where(mag > force_cap, force * (force_cap / mag), force)

        # Sample auxiliary fields with force-bias shift.
        noise = jax.random.normal(key_aux, shape=force.shape, dtype=force.real.dtype)
        aux = noise + force

        # Force-bias contribution to the weight.
        logw_fb = (0.5 * jnp.sum(force * force, axis=-1) - jnp.sum(aux * force, axis=-1)
                   - step * jnp.sum(aux * vbar0, axis=-1)).real

        # Propagate walkers with Trotterized propagator.
        prop_one = lambda w, a: _propagate_one(hmf, vhs_raw, step, expm_apply, dt, w, a)
        walkers_new = jax.vmap(prop_one, in_axes=(0, 0))(state.walkers, aux)

        # Overlap ratio for phaseless constraint.
        o_new = jax.vmap(lambda pt, lw, w: _compute_overlap_from_phi(pt, lw, hamiltonian, w))(
            phi_t, logw_t, walkers_new)

        ratio = o_new / (o_old + 1e-12)
        phase_af = jnp.angle(ratio)
        log_abs_ratio = jnp.log(jnp.abs(ratio) + 1e-12)

        # Base log-weight update before phase constraint.
        logw_base = state.logw + logw_fb + dt * state.et_shift + log_abs_ratio

        # Update trial-path HMC state.
        keys = jax.random.split(key_hmc, state.logw.shape[0])
        sample_one = lambda k, w, s: sampler.sample(k, w, s)
        y_state_new, _ = jax.vmap(sample_one, in_axes=(0, 0, 0))(keys, walkers_new, state.y_state)

        if phase_metro:
            # Phase-Metro correction via overlap phase change after HMC update.
            flat_fields_new = y_state_new[0]
            fields_new = _unravel_batched(unravel_fn, flat_fields_new, (state.logw.shape[0], n_chains))
            o_after = jax.vmap(lambda w, f: _compute_overlap_from_fields(trial, trial_params, w, f),
                               in_axes=(0, 0))(walkers_new, fields_new)
            phase_metro_val = jnp.angle(o_after / (o_new + 1e-12))
            phase_whole = phase_af + phase_metro_val
        else:
            phase_whole = phase_af

        # Phaseless constraint.
        cos_phase = jnp.cos(phase_whole)
        alive = (cos_phase > 0) & state.alive
        logw_new = logw_base + jnp.log(jnp.clip(cos_phase, 1e-12))
        logw_new = jnp.where(alive, logw_new, -1e30)

        # Population control (cap + SR) at configured frequency.
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

        # Optional QR stabilization.
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

        # Optional ET shift feedback from population weights.
        et_shift = state.et_shift
        if et_update and et_gamma > 0:
            max_log = jnp.max(logw_new)
            weights = jnp.exp(logw_new - max_log)
            wsum = jnp.sum(weights)
            et_corr = -et_gamma * (jnp.log(jnp.clip(wsum, 1e-30)) + max_log - jnp.log(weights.shape[0])) / dt
            et_shift = et_shift + et_corr

        new_state = AFQMCState(state.step + 1, walkers_new, logw_new, y_state_new, key, alive, et_shift)
        stats = {"eloc": eloc.real, "phase": phase_whole, "alive": alive, "et": et_shift}
        return new_state, stats

    return step_fn


def run(cfg):
    """Run a full AFQMC projection using VAFQMC trial parameters."""
    logging.basicConfig(force=True, format="# [%(asctime)s] %(levelname)s: %(message)s")
    logger = logging.getLogger("afqmc")
    logger.setLevel(getattr(logging, cfg.log.level.upper()))

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
    trial_params = _extract_params(params_raw)

    bg_flag = cfg.afqmc.force_background
    bg_wfn = hamiltonian.wfn0 if bg_flag and str(bg_flag).lower() == "hf" else None
    hmf, vhs_raw, vbar0, step, expm_apply, enuc = _build_ops(
        hamiltonian, cfg.afqmc.dt, cfg.afqmc.expm_option, bg_wfn)

    fields_shape = trial.fields_shape()
    fsize, unravel_fn = ravel_shape(fields_shape)

    sampler = _make_trial_sampler(trial, trial_params, hamiltonian, fields_shape, cfg.afqmc.hmc)

    n_walkers = cfg.afqmc.walkers
    walkers = _stack_walkers(hamiltonian.wfn0, n_walkers)
    logw = jnp.zeros((n_walkers,))
    alive = jnp.ones((n_walkers,), dtype=bool)

    key = jax.random.PRNGKey(cfg.seed or 0)
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
            per_walker = lambda w, f: _compute_force_and_overlap(
                trial, trial_params, hamiltonian, vhs_raw, vbar0, w, f
            )[-1]
            eloc = jax.vmap(per_walker, in_axes=(0, 0))(walkers, fields)
            logger.info(f"burn-in mean_eloc={float(jnp.mean(eloc)):.6f}")

    state = AFQMCState(0, walkers, logw, y_state, key, alive, jnp.array(cfg.afqmc.ET))
    step_fn = make_afqmc_step(
        trial, trial_params, hamiltonian, hmf, vhs_raw, vbar0, step,
        expm_apply, sampler, unravel_fn, {
            "dt": cfg.afqmc.dt,
            "force_cap": cfg.afqmc.force_cap,
            "stabilize_step": cfg.afqmc.stabilize_step,
            "hmc": {"n_chains": cfg.afqmc.hmc.n_chains},
            "phase_metro": cfg.afqmc.phase_metro,
            "pop_control": cfg.afqmc.pop_control,
            "et_update": cfg.afqmc.et_update.enabled,
            "et_gamma": cfg.afqmc.et_update.gamma,
        },
    )

    def scan_body(carry, _):
        new_state, stats = step_fn(carry)
        return new_state, stats

    steps = int(cfg.afqmc.steps)
    run_scan = jax.jit(lambda st: lax.scan(scan_body, st, xs=None, length=steps))
    state, stats = run_scan(state)

    if cfg.log.stat_freq > 0:
        stat_freq = int(cfg.log.stat_freq)
        for ii in range(0, steps, stat_freq):
            mean_eloc = jnp.mean(stats["eloc"][ii])
            alive_count = jnp.sum(stats["alive"][ii])
            et_now = stats["et"][ii]
            logger.info(f"step={ii} eloc={mean_eloc:.6f} alive={int(alive_count)} ET={float(et_now):.6f}")

    return state
