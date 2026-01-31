"""Walker state and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax
from jax import numpy as jnp
from jax import tree_util

from ..propagator import orthonormalize
from .afqmc_utils import Array, Wfn, calc_slov_batch, _require_spin_det, _is_custom_trial


@dataclass
class AFQMCState:
    walkers: Any
    weights: Array
    sign: Array
    logov: Array
    key: Array
    e_estimate: Array
    pop_control_shift: Array


# Register AFQMCState as a PyTree node for JAX jit/scan usage.
tree_util.register_pytree_node(
    AFQMCState,
    lambda s: (
        (
            s.walkers,
            s.weights,
            s.sign,
            s.logov,
            s.key,
            s.e_estimate,
            s.pop_control_shift,
        ),
        None,
    ),
    lambda aux, children: AFQMCState(*children),
)


def init_walkers(
    trial: Any,
    n_walkers: int,
    key: Array,
    noise: float = 0.0,
) -> Tuple[Any, Array]:
    if _is_custom_trial(trial):
        if hasattr(trial, "init_walkers"):
            return trial.init_walkers(n_walkers, key, noise), key
        if hasattr(trial, "get_init_walkers"):
            return trial.get_init_walkers(n_walkers, key, noise), key
        raise ValueError("Custom trial must provide init_walkers/get_init_walkers.")

    _require_spin_det(trial)
    w_up, w_dn = trial
    w_up = jnp.broadcast_to(w_up, (n_walkers,) + w_up.shape)
    w_dn = jnp.broadcast_to(w_dn, (n_walkers,) + w_dn.shape)

    if noise > 0.0:
        key, k1, k2 = jax.random.split(key, 3)
        w_up = w_up + noise * jax.random.normal(k1, w_up.shape)
        w_dn = w_dn + noise * jax.random.normal(k2, w_dn.shape)

    (w_up, w_dn), _ = orthonormalize((w_up, w_dn))
    return (w_up, w_dn), key


def maybe_orthonormalize(trial: Any, state: AFQMCState, step: int, interval: int) -> AFQMCState:
    if interval <= 0 or (step + 1) % interval != 0:
        return state

    if _is_custom_trial(trial) and hasattr(trial, "orthonormalize_walkers"):
        walkers = trial.orthonormalize_walkers(state.walkers)
    else:
        _require_spin_det(state.walkers)
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
    )


def stochastic_reconfiguration(trial: Any, state: AFQMCState, key: Array) -> AFQMCState:
    if _is_custom_trial(trial) and hasattr(trial, "stochastic_reconfiguration"):
        walkers, weights = trial.stochastic_reconfiguration(state.walkers, state.weights, key)
        sign, logov = calc_slov_batch(trial, walkers)
        return AFQMCState(
            walkers=walkers,
            weights=weights,
            sign=sign,
            logov=logov,
            key=state.key,
            e_estimate=state.e_estimate,
            pop_control_shift=state.pop_control_shift,
        )

    n_walkers = state.weights.shape[0]
    weights = jnp.maximum(state.weights, 0.0)
    wsum = jnp.maximum(jnp.sum(weights), 1.0e-12)
    probs = weights / wsum
    cdf = jnp.cumsum(probs)
    u0 = jax.random.uniform(key, ())
    positions = (u0 + jnp.arange(n_walkers)) / n_walkers
    idx = jnp.searchsorted(cdf, positions)
    idx = jnp.clip(idx, 0, n_walkers - 1)

    w_up, w_dn = state.walkers
    w_up = w_up[idx]
    w_dn = w_dn[idx]
    walkers = (w_up, w_dn)

    new_weights = jnp.ones_like(weights) * (wsum / n_walkers)
    sign, logov = calc_slov_batch(trial, walkers)
    return AFQMCState(
        walkers=walkers,
        weights=new_weights,
        sign=sign,
        logov=logov,
        key=state.key,
        e_estimate=state.e_estimate,
        pop_control_shift=state.pop_control_shift,
    )
