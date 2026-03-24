"""Single-host multi-GPU helpers for walker-sharded AFQMC."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
from jax import lax, numpy as jnp
from jax.tree_util import tree_map

from ..walker import AFQMCState, stochastic_reconfiguration

PMAP_AXIS_NAME = "afqmc_device"

_RUNTIME_STATE_BATCH_KEYS = (
    "pool_state",
    "pool_fields_tree",
    "pool_logsw",
    "pool_bra",
    "pool_log_abs",
    "pool_phase",
    "walker_fields",
)


def require_local_devices(n_walkers: int) -> tuple[int, int, Sequence[Any]]:
    n_devices = int(jax.local_device_count())
    if n_devices <= 1:
        raise ValueError("multi_gpu walker distribution requires at least 2 visible local devices.")
    if int(n_walkers) % n_devices != 0:
        raise ValueError(
            f"n_walkers={int(n_walkers)} must be divisible by visible local devices={n_devices}."
        )
    n_local = int(n_walkers) // n_devices
    return n_devices, n_local, tuple(jax.local_devices()[:n_devices])


def shard_local_pytrees(local_values: Sequence[Any], devices: Sequence[Any]) -> Any:
    return tree_map(
        lambda *xs: jax.device_put_sharded(list(xs), devices),
        *local_values,
    )


def host_replicated_value(x: Any) -> Any:
    return jax.device_get(x)[0]


def _host_unshard_batch(x: Any) -> Any:
    arr = jnp.asarray(jax.device_get(x))
    if arr.ndim < 2:
        return arr
    return arr.reshape((arr.shape[0] * arr.shape[1],) + arr.shape[2:])


def _host_gather_runtime_state(runtime_state: Any) -> Any:
    if runtime_state is None:
        return None
    gathered = {"rng": host_replicated_value(runtime_state["rng"])}
    for key in _RUNTIME_STATE_BATCH_KEYS:
        gathered[key] = tree_map(_host_unshard_batch, runtime_state[key])
    return gathered


def host_gather_state(state: AFQMCState) -> AFQMCState:
    return AFQMCState(
        walkers=tree_map(_host_unshard_batch, state.walkers),
        weights=_host_unshard_batch(state.weights),
        sign=_host_unshard_batch(state.sign),
        logov=_host_unshard_batch(state.logov),
        key=host_replicated_value(state.key),
        e_estimate=host_replicated_value(state.e_estimate),
        pop_control_shift=host_replicated_value(state.pop_control_shift),
        walker_fields=_host_unshard_batch(state.walker_fields),
        trial_state=_host_gather_runtime_state(state.trial_state),
    )


def _all_gather_flat(x: Any, axis_name: str) -> Any:
    gathered = lax.all_gather(x, axis_name)
    if gathered.ndim < 2:
        return gathered
    return gathered.reshape((gathered.shape[0] * gathered.shape[1],) + gathered.shape[2:])


def _slice_local_batch(x: Any, start: Any, size: int) -> Any:
    if getattr(x, "ndim", 0) == 0:
        return x
    if int(x.shape[0]) == 0:
        return x
    return lax.dynamic_slice_in_dim(x, start, size, axis=0)


def _gather_runtime_state(runtime_state: Any, axis_name: str) -> Any:
    if runtime_state is None:
        return None
    gathered = {"rng": lax.all_gather(runtime_state["rng"], axis_name)[0]}
    for key in _RUNTIME_STATE_BATCH_KEYS:
        gathered[key] = tree_map(lambda x: _all_gather_flat(x, axis_name), runtime_state[key])
    return gathered


def _slice_runtime_state(runtime_state: Any, axis_name: str, n_local: int) -> Any:
    if runtime_state is None:
        return None
    axis_idx = lax.axis_index(axis_name)
    start = axis_idx * n_local
    sliced = {"rng": jax.random.fold_in(runtime_state["rng"], axis_idx)}
    for key in _RUNTIME_STATE_BATCH_KEYS:
        sliced[key] = tree_map(
            lambda x: _slice_local_batch(x, start, n_local),
            runtime_state[key],
        )
    return sliced


def distributed_stochastic_reconfiguration(
    trial: Any,
    state: AFQMCState,
    axis_name: str = PMAP_AXIS_NAME,
) -> AFQMCState:
    n_local = int(state.weights.shape[0])
    axis_idx = lax.axis_index(axis_name)
    start = axis_idx * n_local
    global_key = lax.all_gather(state.key, axis_name)[0]

    global_state = AFQMCState(
        walkers=tree_map(lambda x: _all_gather_flat(x, axis_name), state.walkers),
        weights=_all_gather_flat(state.weights, axis_name),
        sign=_all_gather_flat(state.sign, axis_name),
        logov=_all_gather_flat(state.logov, axis_name),
        key=global_key,
        e_estimate=state.e_estimate,
        pop_control_shift=state.pop_control_shift,
        walker_fields=_all_gather_flat(state.walker_fields, axis_name),
        trial_state=_gather_runtime_state(state.trial_state, axis_name),
    )
    global_state = stochastic_reconfiguration(trial, global_state, global_key)

    return AFQMCState(
        walkers=tree_map(
            lambda x: _slice_local_batch(x, start, n_local),
            global_state.walkers,
        ),
        weights=_slice_local_batch(global_state.weights, start, n_local),
        sign=_slice_local_batch(global_state.sign, start, n_local),
        logov=_slice_local_batch(global_state.logov, start, n_local),
        key=jax.random.fold_in(global_key, axis_idx),
        e_estimate=global_state.e_estimate,
        pop_control_shift=global_state.pop_control_shift,
        walker_fields=_slice_local_batch(global_state.walker_fields, start, n_local),
        trial_state=_slice_runtime_state(global_state.trial_state, axis_name, n_local),
    )


__all__ = [
    "PMAP_AXIS_NAME",
    "distributed_stochastic_reconfiguration",
    "host_gather_state",
    "host_replicated_value",
    "require_local_devices",
    "shard_local_pytrees",
]
