"""Internal helper methods for stochastic VAFQMC trial."""

from __future__ import annotations

from typing import Any

import jax
from jax import numpy as jnp
from jax.tree_util import tree_leaves, tree_map

from ...hamiltonian import _has_spin, calc_rdm, calc_slov
from ...sampler import make_batched
from ..afqmc_utils import (
    _spin_sum_rdm,
    gaussian_logdens,
    mix_overlap_terms as _mix_overlap_terms,
)

Array = jnp.ndarray


def _build_logdens(self):
    if self.sampling_target in ("gaussian", "normal", "std"):
        return gaussian_logdens

    if self.sampling_target in ("coeff_overlap", "reference_overlap"):
        def _logdens(params: Any, fields: Any) -> Array:
            ansatz_params = params[0] if isinstance(params, (tuple, list)) else params
            bra, bra_lw = self.ansatz.apply(ansatz_params, fields)
            _, logov = calc_slov(bra, self.reference_wfn)
            logd = jnp.real(logov + bra_lw)
            return jnp.where(jnp.isfinite(logd), logd, self.logdens_floor)

        return _logdens

    def _logdens(params: Any, fields: Any) -> Array:
        if isinstance(params, (tuple, list)) and len(params) == 2:
            ansatz_params, walker = params
        else:
            ansatz_params, walker = params, self.reference_wfn
        bra, bra_lw = self.ansatz.apply(ansatz_params, fields)
        _, logov = calc_slov(bra, walker)
        logd = jnp.real(logov + bra_lw)
        return jnp.where(jnp.isfinite(logd), logd, self.logdens_floor)

    return _logdens


def _collapse_rdm(self, rdm: Array) -> Array:
    if rdm.ndim == 2 and rdm.shape[-1] == 2 * self.nbasis:
        blk = rdm.reshape(2, self.nbasis, 2, self.nbasis).swapaxes(1, 2)
        return blk[0, 0] + blk[1, 1]
    return _spin_sum_rdm(rdm)


def _fields_to_flat(self, fields: Any) -> Array:
    return jnp.concatenate([jnp.ravel(x) for x in tree_leaves(fields)], axis=0)


def _fields_tree_to_flat(self, fields_tree: Any) -> Array:
    return jax.vmap(jax.vmap(self._fields_to_flat))(fields_tree)


def _sample_walkers_from_trial(self, n_walkers: int, key: Array) -> Any:
    """Initialize walkers by sampling ansatz outputs from trial distribution."""
    walker_sampler = make_batched(self._sampler, n_walkers, concat=False)
    sampler_params = (self.params, self.reference_wfn)

    key, init_key = jax.random.split(key)
    state = walker_sampler.init(init_key, sampler_params)
    if self.init_walkers_burn_in > 0:
        key, burn_key = jax.random.split(key)
        state = walker_sampler.burn_in(
            burn_key,
            sampler_params,
            state,
            self.init_walkers_burn_in,
        )
    key, sample_key = jax.random.split(key)
    _, (fields, _logsw) = walker_sampler.sample(sample_key, sampler_params, state)
    walkers, _ = jax.vmap(lambda f: self.ansatz.apply(self.params, f))(fields)
    if not _has_spin(walkers):
        raise ValueError("init_walkers_from_trial requires spin-separated ansatz output.")
    return walkers


def _evaluate_pool_bra(self, fields: Any, logsw: Array) -> tuple[Any, Array]:
    eval_ansatz = jax.vmap(jax.vmap(lambda f: self.ansatz.apply(self.params, f)))
    bra_samples, bra_logw = eval_ansatz(fields)
    return bra_samples, (bra_logw - logsw)


def _update_pool_cache(self, fields: Any, logsw: Array) -> None:
    self._pool_fields_tree = fields
    self._pool_logsw = logsw
    self._pool_bra, logcoef = self._evaluate_pool_bra(fields, logsw)
    self._pool_log_abs = jnp.real(logcoef)
    self._pool_phase = jnp.exp(1.0j * jnp.imag(logcoef))
    self.walker_fields = self._fields_tree_to_flat(fields)


def _init_pool_state(self, walkers: Any, burn_steps: int):
    self._rng, init_key = jax.random.split(self._rng)
    init_keys = jax.random.split(init_key, int(tree_leaves(walkers)[0].shape[0]))
    state = jax.vmap(lambda k, w: self._pool_sampler.init(k, (self.params, w)))(init_keys, walkers)

    if burn_steps > 0:
        self._rng, burn_key = jax.random.split(self._rng)
        burn_keys = jax.random.split(burn_key, int(tree_leaves(walkers)[0].shape[0]))
        state = jax.vmap(
            lambda k, w, st: self._pool_sampler.burn_in(k, (self.params, w), st, burn_steps)
        )(burn_keys, walkers, state)
    return state


def _sample_pool_step(self, walkers: Any, state: Any):
    self._rng, sample_key = jax.random.split(self._rng)
    keys = jax.random.split(sample_key, self._n_walkers)
    sample_one = lambda key, walker, st: self._pool_sampler.sample(key, (self.params, walker), st)
    return jax.vmap(sample_one)(keys, walkers, state)


def _ensure_bound_pool(self, walkers: Any) -> None:
    walkers = tree_map(jnp.asarray, walkers)
    nw = int(tree_leaves(walkers)[0].shape[0])
    bad = (
        self._pool_state is None
        or self._pool_logsw is None
        or self._n_walkers != nw
        or int(self._pool_logsw.shape[1]) != self.n_samples
    )
    if bad:
        self.bind_walkers(walkers, reinit=True)
    else:
        self._bound_walkers = walkers


def _calc_overlap_bundle_from_cache(
    self,
    walkers: Any,
    pool_bra: Any,
    pool_log_abs: Array,
    pool_phase: Array,
) -> tuple[Array, Array, Array]:
    def one_walker(walker, bra_samples, log_abs, phase):
        sample_sign, sample_logov = jax.vmap(lambda bra: calc_slov(bra, walker))(bra_samples)
        return _mix_overlap_terms(sample_sign, sample_logov, log_abs, phase)

    return jax.vmap(one_walker)(
        walkers,
        pool_bra,
        pool_log_abs,
        pool_phase,
    )


def _get_local_energy_fn(self, hamil: Any):
    key = id(hamil)
    fn = self._local_energy_fns.get(key)
    if fn is None:
        def local_energy_fn(walkers, pool_bra, mix_weights):
            def one_walker(walker, bra_samples, weights):
                es = jax.vmap(lambda bra: hamil.local_energy(bra, walker))(bra_samples)
                return jnp.einsum("s,s->", weights, es)

            return jax.vmap(one_walker)(walkers, pool_bra, mix_weights)

        fn = jax.jit(local_energy_fn)
        self._local_energy_fns[key] = fn
    return fn


def _get_force_bias_fn(self, prop_data: Any):
    key = id(prop_data)
    fn = self._force_bias_fns.get(key)
    if fn is None:
        def force_bias_fn(walkers, pool_bra, mix_weights):
            def one_walker(walker, bra_samples, weights):
                rdms = jax.vmap(lambda bra: self._collapse_rdm(calc_rdm(bra, walker)))(bra_samples)
                rdm_mix = jnp.einsum("s,spq->pq", weights, rdms)
                return jnp.einsum("kpq,pq->k", prop_data.vhs, rdm_mix)

            return jax.vmap(one_walker)(walkers, pool_bra, mix_weights)

        fn = jax.jit(force_bias_fn)
        self._force_bias_fns[key] = fn
    return fn


__all__ = [
    "_build_logdens",
    "_calc_overlap_bundle_from_cache",
    "_collapse_rdm",
    "_ensure_bound_pool",
    "_evaluate_pool_bra",
    "_fields_to_flat",
    "_fields_tree_to_flat",
    "_get_force_bias_fn",
    "_get_local_energy_fn",
    "_init_pool_state",
    "_sample_pool_step",
    "_sample_walkers_from_trial",
    "_update_pool_cache",
]
