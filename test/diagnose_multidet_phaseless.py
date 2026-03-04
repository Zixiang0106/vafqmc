"""Diagnose stochastic multi-det AFQMC instability without modifying kernels.

This script runs custom-trial propagation step-by-step and prints diagnostics:
- non-finite force-bias walkers
- non-finite RDM samples (all vs active samples with |mix_weight| > eps)
- sign-ratio negative walkers that survive after propagation
- weight summary (min/max/mean/sum)
"""

from __future__ import annotations

import argparse
from typing import Any

import jax
from jax import numpy as jnp

from hafqmc.afqmc.afqmc import _build_trial_from_config, _runtime_cfg, _to_cfg
from hafqmc.afqmc.afqmc_config import AFQMCConfig
from hafqmc.afqmc.afqmc_utils import apply_trotter, calc_force_bias, calc_slov_batch, load_hamiltonian
from hafqmc.afqmc.driver.custom import (
    _build_initial_state_custom,
    _orthonormalize_custom,
    _resample_state_custom,
)
from hafqmc.afqmc.walker import AFQMCState
from hafqmc.hamiltonian import calc_rdm


def _trial_diag(trial: Any, state: Any, prop_data: Any, weight_eps: float):
    ts = state.trial_state
    if ts is None:
        return None
    if not hasattr(trial, "_overlap_bundle_from_state"):
        return None
    if not hasattr(trial, "_collapse_rdm"):
        return None

    sign, logov, mix_weights = trial._overlap_bundle_from_state(state.walkers, ts)
    if hasattr(trial, "calc_force_bias_state"):
        fb = trial.calc_force_bias_state(state.walkers, prop_data, ts)
    else:
        fb = None

    pool_bra = ts["pool_bra"]

    def one_walker(walker, bra_samples, wmix):
        rdms = jax.vmap(lambda bra: trial._collapse_rdm(calc_rdm(bra, walker)))(bra_samples)
        finite_rdm = jnp.all(jnp.isfinite(rdms), axis=(1, 2))
        active = jnp.abs(wmix) > weight_eps
        bad_active = jnp.any(active & (~finite_rdm))
        bad_any = jnp.any(~finite_rdm)
        return (
            bad_active,
            bad_any,
            jnp.sum(active),
            jnp.sum(~finite_rdm),
            jnp.min(jnp.real(wmix)),
            jnp.max(jnp.real(wmix)),
        )

    bad_active, bad_any, n_active, n_bad_rdm, wmix_min, wmix_max = jax.vmap(one_walker)(
        state.walkers, pool_bra, mix_weights
    )

    out = {
        "n_walkers": int(state.weights.shape[0]),
        "n_logov_nonfinite": int(jnp.sum(~jnp.isfinite(logov))),
        "n_bad_rdm_any": int(jnp.sum(bad_any)),
        "n_bad_rdm_active": int(jnp.sum(bad_active)),
        "n_active_samples_min": int(jnp.min(n_active)),
        "n_active_samples_max": int(jnp.max(n_active)),
        "n_bad_rdm_samples_max": int(jnp.max(n_bad_rdm)),
        "wmix_min": float(jnp.min(wmix_min)),
        "wmix_max": float(jnp.max(wmix_max)),
    }

    if fb is not None:
        fb_finite = jnp.all(jnp.isfinite(fb), axis=1)
        out["n_force_bias_nonfinite"] = int(jnp.sum(~fb_finite))
        out["fb_abs_max"] = float(jnp.max(jnp.abs(fb)))
    else:
        out["n_force_bias_nonfinite"] = -1
        out["fb_abs_max"] = float("nan")

    return out


def _probe_step_custom(hamil: Any, trial: Any, state: AFQMCState, prop_data: Any, cfg: Any):
    """Replica of one custom propagation step with extra diagnostics."""
    n_walkers = state.weights.shape[0]
    nchol = prop_data.vhs.shape[0]
    trial_state = state.trial_state

    key, subkey = jax.random.split(state.key)
    fields = jax.random.normal(subkey, (n_walkers, nchol), dtype=jnp.real(prop_data.vhs).dtype)

    if trial_state is not None and hasattr(trial, "calc_force_bias_state"):
        force_bias = trial.calc_force_bias_state(state.walkers, prop_data, trial_state)
    else:
        force_bias = calc_force_bias(hamil, trial, state.walkers, prop_data)
    fb_finite = jnp.all(jnp.isfinite(force_bias), axis=1)

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
    imp_ph_raw = jnp.abs(imp) * jnp.cos(theta)
    imp_ph = jnp.where(jnp.isnan(imp_ph_raw), 0.0, imp_ph_raw)
    imp_ph = jnp.where(imp_ph < 0.0, 0.0, imp_ph)
    imp_ph = jnp.where(imp_ph < cfg.min_weight, 0.0, imp_ph)
    imp_ph = jnp.where(imp_ph > cfg.max_weight, 0.0, imp_ph)

    sign_new, logov_new = sign_prop, logov_prop
    walker_fields = state.walker_fields
    handoff = jnp.ones_like(imp_ph)
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

    state_new = AFQMCState(
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

    should_kill_raw = (
        (~jnp.isfinite(imp_ph_raw))
        | (~jnp.isfinite(sign_prop))
        | (~jnp.isfinite(logov_prop))
        | (~fb_finite)
        | (imp_ph_raw <= 0.0)
    )
    survived_should_kill = should_kill_raw & (weights > 0.0)

    diag_step = {
        "n_should_kill_raw": int(jnp.sum(should_kill_raw)),
        "n_survived_should_kill": int(jnp.sum(survived_should_kill)),
        "n_imp_ph_raw_nonfinite": int(jnp.sum(~jnp.isfinite(imp_ph_raw))),
        "n_imp_ph_raw_le0": int(jnp.sum(imp_ph_raw <= 0.0)),
        "n_fb_nonfinite": int(jnp.sum(~fb_finite)),
        "n_logov_prop_nonfinite": int(jnp.sum(~jnp.isfinite(logov_prop))),
        "handoff_min": float(jnp.min(handoff)),
        "handoff_max": float(jnp.max(handoff)),
    }
    return state_new, diag_step


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hamil", type=str, default="hamiltonian.pkl")
    p.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint.pkl")
    p.add_argument("--hparams", type=str, default="hparams.yml")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--walkers", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--burn-in", type=int, default=100)
    p.add_argument("--sample-update-steps", type=int, default=1)
    p.add_argument("--pop-freq", type=int, default=5)
    p.add_argument("--ortho-freq", type=int, default=10)
    p.add_argument("--weight-eps", type=float, default=1e-14)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--stop-on-anomaly", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = AFQMCConfig.stochastic_example()
    cfg.seed = int(args.seed)
    cfg.propagation.dt = float(args.dt)
    cfg.propagation.n_walkers = int(args.walkers)
    cfg.propagation.n_eq_steps = 0
    cfg.propagation.n_blocks = 1
    cfg.propagation.n_block_steps = 1
    cfg.propagation.ortho_freq = int(args.ortho_freq)
    cfg.log.block_freq = 1
    cfg.log.pop_control_stats = False
    cfg.pop_control.resample = True
    cfg.pop_control.freq = int(args.pop_freq)

    cfg.trial_type = "stochastic"
    cfg.stochastic_trial.checkpoint = str(args.checkpoint)
    cfg.stochastic_trial.hparams_path = str(args.hparams)
    cfg.stochastic_trial.n_samples = int(args.samples)
    cfg.stochastic_trial.burn_in = int(args.burn_in)
    cfg.stochastic_trial.sample_update_steps = int(args.sample_update_steps)

    cfg_rt = _runtime_cfg(_to_cfg(cfg))
    hamil = load_hamiltonian(args.hamil)
    trial = _build_trial_from_config(hamil, cfg_rt)
    prop_data, state = _build_initial_state_custom(hamil, trial, cfg_rt)

    print(
        f"diag init: walkers={cfg_rt.n_walkers} steps={args.steps} "
        f"pop_freq={cfg_rt.pop_control_freq} ortho_freq={cfg_rt.ortho_freq}"
    )
    print(
        f"init weights: min={float(jnp.min(state.weights)):.3e} "
        f"max={float(jnp.max(state.weights)):.3e} sum={float(jnp.sum(state.weights)):.3e}"
    )

    for step in range(int(args.steps)):
        diag_pre = _trial_diag(trial, state, prop_data, float(args.weight_eps))
        prev_sign = state.sign
        prev_logov = state.logov

        state, diag_step = _probe_step_custom(hamil, trial, state, prop_data, cfg_rt)

        if cfg_rt.ortho_freq > 0 and ((step + 1) % cfg_rt.ortho_freq == 0):
            state = _orthonormalize_custom(trial, state, jnp.asarray(True))

        if cfg_rt.resample and cfg_rt.pop_control_freq > 0 and ((step + 1) % cfg_rt.pop_control_freq == 0):
            state = _resample_state_custom(trial, state)

        valid = jnp.isfinite(prev_logov) & jnp.isfinite(state.logov) & (jnp.abs(prev_sign) > 0)
        sign_ratio = jnp.where(valid, state.sign / prev_sign, 0.0 + 0.0j)
        neg_ratio = jnp.real(sign_ratio) < 0.0
        survive_neg = neg_ratio & (state.weights > 0.0)

        w = state.weights
        msg = (
            f"step {step+1:4d} "
            f"w(min/max/mean/sum)=({float(jnp.min(w)):.3e}/{float(jnp.max(w)):.3e}/"
            f"{float(jnp.mean(w)):.3e}/{float(jnp.sum(w)):.3e}) "
            f"survive_neg={int(jnp.sum(survive_neg))} "
            f"kill_raw={diag_step['n_should_kill_raw']} "
            f"survive_kill_raw={diag_step['n_survived_should_kill']}"
        )
        if diag_pre is not None:
            msg += (
                f" fb_nonfinite={diag_pre['n_force_bias_nonfinite']}"
                f" bad_rdm_active={diag_pre['n_bad_rdm_active']}"
                f" bad_rdm_any={diag_pre['n_bad_rdm_any']}"
                f" logov_nonfinite={diag_pre['n_logov_nonfinite']}"
                f" wmix[min,max]=({diag_pre['wmix_min']:.3e},{diag_pre['wmix_max']:.3e})"
            )

        if (step + 1) % max(int(args.log_every), 1) == 0:
            print(msg)

        anomaly = (
            (diag_pre is not None and diag_pre["n_force_bias_nonfinite"] > 0)
            or (diag_pre is not None and diag_pre["n_bad_rdm_active"] > 0)
            or (diag_step["n_survived_should_kill"] > 0)
            or (not bool(jnp.isfinite(jnp.sum(state.weights))))
        )
        if anomaly and args.stop_on_anomaly:
            print("ANOMALY detected, stop early.")
            break


if __name__ == "__main__":
    main()
