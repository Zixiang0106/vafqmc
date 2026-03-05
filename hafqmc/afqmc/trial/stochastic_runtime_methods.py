"""Runtime-state and public trial API methods for stochastic VAFQMC trial."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import jax
from jax import lax, numpy as jnp
from jax.tree_util import tree_leaves, tree_map

from ...hamiltonian import _has_spin
from ...propagator import orthonormalize
from ..afqmc_utils import phaseless_from_ratio as _phaseless_from_ratio

Array = jnp.ndarray
RuntimeState = Dict[str, Any]


def _pack_runtime_state(
    self,
    *,
    rng: Any,
    pool_state: Any,
    pool_fields_tree: Any,
    pool_logsw: Array,
    pool_bra: Any,
    pool_log_abs: Array,
    pool_phase: Array,
    walker_fields: Optional[Array] = None,
) -> RuntimeState:
    if walker_fields is None:
        walker_fields = self._fields_tree_to_flat(pool_fields_tree)
    return {
        "rng": rng,
        "pool_state": pool_state,
        "pool_fields_tree": pool_fields_tree,
        "pool_logsw": pool_logsw,
        "pool_bra": pool_bra,
        "pool_log_abs": pool_log_abs,
        "pool_phase": pool_phase,
        "walker_fields": walker_fields,
    }


def _require_runtime_state(self, walkers: Any) -> RuntimeState:
    self._ensure_bound_pool(walkers)
    runtime_state = self.export_runtime_state()
    if runtime_state is None:
        raise ValueError("trial runtime_state is not initialized.")
    return runtime_state


def _overlap_bundle_from_state(
    self,
    walkers: Any,
    runtime_state: RuntimeState,
) -> tuple[Array, Array, Array]:
    return self._calc_overlap_bundle_from_cache(
        walkers,
        runtime_state["pool_bra"],
        runtime_state["pool_log_abs"],
        runtime_state["pool_phase"],
    )


def _cache_from_samples(self, fields: Any, logsw: Array) -> tuple[Any, Array, Array]:
    bra, logcoef = self._evaluate_pool_bra(fields, logsw)
    return bra, jnp.real(logcoef), jnp.exp(1.0j * jnp.imag(logcoef))


def export_runtime_state(self) -> Optional[RuntimeState]:
    if self._pool_state is None or self._pool_logsw is None:
        return None
    return self._pack_runtime_state(
        rng=self._rng,
        pool_state=self._pool_state,
        pool_fields_tree=self._pool_fields_tree,
        pool_logsw=self._pool_logsw,
        pool_bra=self._pool_bra,
        pool_log_abs=self._pool_log_abs,
        pool_phase=self._pool_phase,
        walker_fields=self.walker_fields,
    )


def import_runtime_state(self, runtime_state: Optional[RuntimeState]) -> None:
    if runtime_state is None:
        return
    self._rng = runtime_state["rng"]
    self._pool_state = runtime_state["pool_state"]
    self._pool_fields_tree = runtime_state["pool_fields_tree"]
    self._pool_logsw = runtime_state["pool_logsw"]
    self._pool_bra = runtime_state["pool_bra"]
    self._pool_log_abs = runtime_state["pool_log_abs"]
    self._pool_phase = runtime_state["pool_phase"]
    self.walker_fields = runtime_state["walker_fields"]
    self._n_walkers = int(self._pool_logsw.shape[0])


def init_runtime_state(
    self,
    walkers: Any,
    *,
    key: Optional[Array] = None,
    seed: Optional[int] = None,
    burn_in: Optional[int] = None,
    reinit: bool = True,
) -> RuntimeState:
    self.bind_walkers(
        walkers,
        key=key,
        seed=seed,
        burn_in=burn_in,
        reinit=reinit,
    )
    state = self.export_runtime_state()
    if state is None:
        raise ValueError("Failed to initialize runtime_state.")
    return state


def on_walkers_updated_state(
    self,
    walkers: Any,
    runtime_state: Optional[RuntimeState],
) -> RuntimeState:
    walkers = tree_map(jnp.asarray, walkers)
    nw = int(tree_leaves(walkers)[0].shape[0])
    if runtime_state is None:
        return self.init_runtime_state(walkers, reinit=True)
    if int(runtime_state["pool_logsw"].shape[0]) != nw:
        return self.init_runtime_state(walkers, reinit=True)
    return runtime_state


def get_rdm1(self) -> Array:
    return self.rdm1


def init_walkers(self, n_walkers: int, key: Array, noise: float = 0.0) -> Any:
    if self.init_walkers_from_trial:
        walkers = self._sample_walkers_from_trial(n_walkers, key)
    else:
        if not _has_spin(self.reference_wfn):
            raise ValueError("VAFQMCTrial requires spin-separated reference_wfn for AFQMC walkers.")
        w_up, w_dn = self.reference_wfn
        w_up = jnp.broadcast_to(w_up, (n_walkers,) + w_up.shape)
        w_dn = jnp.broadcast_to(w_dn, (n_walkers,) + w_dn.shape)
        walkers = (w_up, w_dn)

    if noise > 0.0:
        key, k1, k2 = jax.random.split(key, 3)
        w_up, w_dn = walkers
        w_up = w_up + noise * jax.random.normal(k1, w_up.shape)
        w_dn = w_dn + noise * jax.random.normal(k2, w_dn.shape)
        walkers = (w_up, w_dn)

    walkers, _ = orthonormalize(walkers)
    return walkers


def orthonormalize_walkers(self, walkers: Any) -> Any:
    walkers, _ = orthonormalize(walkers)
    return walkers


def on_walkers_updated(self, walkers: Any) -> None:
    walkers = tree_map(jnp.asarray, walkers)
    nw = int(tree_leaves(walkers)[0].shape[0])
    if nw != self._n_walkers or self._pool_state is None:
        self.bind_walkers(walkers, reinit=True)
    else:
        self._bound_walkers = walkers


def bind_walkers(
    self,
    walkers: Any,
    *,
    key: Optional[Array] = None,
    seed: Optional[int] = None,
    burn_in: Optional[int] = None,
    reinit: bool = True,
) -> None:
    walkers = tree_map(jnp.asarray, walkers)
    n_walkers = int(tree_leaves(walkers)[0].shape[0])
    if n_walkers <= 0:
        raise ValueError("walkers must have positive leading batch dimension.")

    if seed is not None:
        self._rng = jax.random.PRNGKey(int(seed))
    if key is not None:
        self._rng = key

    requested_burn = self.burn_in if burn_in is None else int(burn_in)
    if requested_burn < 0:
        raise ValueError("burn_in must be non-negative.")

    need_init = (
        bool(reinit)
        or self._pool_state is None
        or self._pool_logsw is None
        or self._n_walkers != n_walkers
        or int(self._pool_logsw.shape[1]) != self.n_samples
    )
    if need_init:
        if requested_burn > 0:
            n_chains = int(n_walkers * self.n_samples)
            logging.getLogger("hafqmc.afqmc").info(
                "Burning in the sampler of %d chains for %d steps (n_walkers=%d, n_samples=%d).",
                n_chains,
                requested_burn,
                n_walkers,
                self.n_samples,
            )
        self._pool_state = self._init_pool_state(walkers, requested_burn)

    self._n_walkers = n_walkers
    self._bound_walkers = walkers
    self._pool_state, (fields, logsw) = self._sample_pool_step(walkers, self._pool_state)
    self._update_pool_cache(fields, logsw)


def calc_slov(self, walkers: Any) -> tuple[Array, Array]:
    return self.calc_slov_state(walkers, self._require_runtime_state(walkers))


def calc_slov_state(
    self,
    walkers: Any,
    runtime_state: RuntimeState,
) -> tuple[Array, Array]:
    walkers = tree_map(jnp.asarray, walkers)
    sign, logov, _ = self._overlap_bundle_from_state(walkers, runtime_state)
    return sign, logov


def calc_local_energy(self, hamil: Any, walkers: Any) -> Array:
    return self.calc_local_energy_state(hamil, walkers, self._require_runtime_state(walkers))


def calc_local_energy_state(
    self,
    hamil: Any,
    walkers: Any,
    runtime_state: RuntimeState,
) -> Array:
    walkers = tree_map(jnp.asarray, walkers)
    _, _, mix_weights = self._overlap_bundle_from_state(walkers, runtime_state)
    fn = self._get_local_energy_fn(hamil)
    chunk = int(getattr(self, "local_energy_chunk_size", 0))
    if chunk <= 0:
        return fn(walkers, runtime_state["pool_bra"], mix_weights)

    n_walkers = int(tree_leaves(walkers)[0].shape[0])
    if chunk >= n_walkers:
        return fn(walkers, runtime_state["pool_bra"], mix_weights)

    # Reduce peak memory by evaluating local energy on walker chunks.
    energies = []
    for start in range(0, n_walkers, chunk):
        end = min(start + chunk, n_walkers)
        w_chunk = tree_map(lambda x: x[start:end], walkers)
        bra_chunk = tree_map(lambda x: x[start:end], runtime_state["pool_bra"])
        mw_chunk = mix_weights[start:end]
        energies.append(fn(w_chunk, bra_chunk, mw_chunk))
    return jnp.concatenate(energies, axis=0)


def calc_force_bias(self, _hamil: Any, walkers: Any, prop_data: Any) -> Array:
    return self.calc_force_bias_state(walkers, prop_data, self._require_runtime_state(walkers))


def calc_force_bias_state(
    self,
    walkers: Any,
    prop_data: Any,
    runtime_state: RuntimeState,
) -> Array:
    walkers = tree_map(jnp.asarray, walkers)
    _, _, mix_weights = self._overlap_bundle_from_state(walkers, runtime_state)
    fn = self._get_force_bias_fn(prop_data)
    return fn(walkers, runtime_state["pool_bra"], mix_weights)


def update_tethered_samples(
    self,
    walkers: Any,
    *,
    n_steps: Optional[int] = None,
    old_sign: Optional[Array] = None,
    old_logov: Optional[Array] = None,
) -> tuple[Array, Array, Array]:
    """Advance each walker's pool and return hand-off correction."""
    runtime_state = self._require_runtime_state(walkers)
    runtime_state, handoff, new_sign, new_logov = self.update_tethered_samples_state(
        walkers,
        runtime_state,
        n_steps=n_steps,
        old_sign=old_sign,
        old_logov=old_logov,
    )
    self.import_runtime_state(runtime_state)
    self._bound_walkers = tree_map(jnp.asarray, walkers)
    return handoff, new_sign, new_logov


def update_tethered_samples_state(
    self,
    walkers: Any,
    runtime_state: RuntimeState,
    *,
    n_steps: Optional[int] = None,
    old_sign: Optional[Array] = None,
    old_logov: Optional[Array] = None,
) -> tuple[RuntimeState, Array, Array, Array]:
    walkers = tree_map(jnp.asarray, walkers)
    if (old_sign is None) or (old_logov is None):
        old_sign, old_logov, _ = self._overlap_bundle_from_state(walkers, runtime_state)
    else:
        old_sign = jnp.asarray(old_sign)
        old_logov = jnp.asarray(old_logov)

    steps = self.sample_update_steps if n_steps is None else int(n_steps)
    if steps < 0:
        raise ValueError("n_steps must be non-negative.")
    if steps == 0:
        ones = jnp.ones((int(tree_leaves(walkers)[0].shape[0]),), dtype=jnp.float64)
        return runtime_state, ones, old_sign, old_logov

    n_walkers = int(tree_leaves(walkers)[0].shape[0])
    sample_one = lambda key, walker, st: self._pool_sampler.sample(key, (self.params, walker), st)

    def step_body(_i, carry):
        rng, st, f, lsw = carry
        rng, sample_key = jax.random.split(rng)
        keys = jax.random.split(sample_key, n_walkers)
        st, (f, lsw) = jax.vmap(sample_one)(keys, walkers, st)
        return (rng, st, f, lsw)

    rng, state_new, fields_new, logsw_new = lax.fori_loop(
        0,
        steps,
        step_body,
        (
            runtime_state["rng"],
            runtime_state["pool_state"],
            runtime_state["pool_fields_tree"],
            runtime_state["pool_logsw"],
        ),
    )
    bra_new, log_abs_new, phase_new = self._cache_from_samples(fields_new, logsw_new)
    new_sign, new_logov, _ = self._calc_overlap_bundle_from_cache(walkers, bra_new, log_abs_new, phase_new)
    valid = jnp.isfinite(old_logov) & jnp.isfinite(new_logov) & (jnp.abs(old_sign) > 0)
    log_ratio = jnp.where(valid, new_logov - old_logov, -jnp.inf)
    sign_ratio = jnp.where(valid, new_sign / old_sign, 0.0 + 0.0j)
    if bool(getattr(self, "debug_print_sign_ratio", False)):
        sign_ratio_flat = sign_ratio.reshape(-1)
        log_ratio_flat = log_ratio.reshape(-1)
        valid_flat = valid.reshape(-1)
        n_ratio = int(sign_ratio_flat.shape[0])

        def _print_sign_ratio(i, _):
            value = sign_ratio_flat[i]
            jax.debug.print(
                "sign_ratio[{i}] = {re} + {im}j, log_ratio={lr} (valid={valid})",
                i=i,
                re=jnp.real(value),
                im=jnp.imag(value),
                lr=log_ratio_flat[i],
                valid=valid_flat[i],
            )
            return _

        _ = lax.fori_loop(0, n_ratio, _print_sign_ratio, None)
    #handoff = _phaseless_from_ratio(sign_ratio * jnp.exp(log_ratio))
    handoff = _phaseless_from_ratio(sign_ratio)

    runtime_state_new = self._pack_runtime_state(
        rng=rng,
        pool_state=state_new,
        pool_fields_tree=fields_new,
        pool_logsw=logsw_new,
        pool_bra=bra_new,
        pool_log_abs=log_abs_new,
        pool_phase=phase_new,
    )
    return runtime_state_new, handoff, new_sign, new_logov


def stochastic_reconfiguration(self, walkers: Any, weights: Array, key: Array):
    """Resample walkers and clone their tethered pools consistently."""
    runtime_state = self._require_runtime_state(walkers)
    walkers_new, new_weights, runtime_state = self.stochastic_reconfiguration_state(
        walkers,
        weights,
        key,
        runtime_state,
    )
    self.import_runtime_state(runtime_state)
    self._bound_walkers = walkers_new
    return walkers_new, new_weights


def stochastic_reconfiguration_state(
    self,
    walkers: Any,
    weights: Array,
    key: Array,
    runtime_state: RuntimeState,
) -> tuple[Any, Array, RuntimeState]:
    n_walkers = int(weights.shape[0])

    pos_weights = jnp.maximum(weights, 0.0)
    wsum = jnp.maximum(jnp.sum(pos_weights), 1.0e-12)
    probs = pos_weights / wsum
    cdf = jnp.cumsum(probs)
    u0 = jax.random.uniform(key, ())
    positions = (u0 + jnp.arange(n_walkers)) / n_walkers
    idx = jnp.searchsorted(cdf, positions)
    idx = jnp.clip(idx, 0, n_walkers - 1)

    walkers_new = tree_map(lambda x: x[idx], walkers)
    runtime_state_new = self._pack_runtime_state(
        rng=runtime_state["rng"],
        pool_state=tree_map(lambda x: x[idx], runtime_state["pool_state"]),
        pool_fields_tree=tree_map(lambda x: x[idx], runtime_state["pool_fields_tree"]),
        pool_logsw=runtime_state["pool_logsw"][idx],
        pool_bra=tree_map(lambda x: x[idx], runtime_state["pool_bra"]),
        pool_log_abs=runtime_state["pool_log_abs"][idx],
        pool_phase=runtime_state["pool_phase"][idx],
        walker_fields=runtime_state["walker_fields"][idx],
    )

    new_weights = jnp.ones_like(weights) * (wsum / n_walkers)
    return walkers_new, new_weights, runtime_state_new


def measure_block_energy(
    self,
    hamil: Any,
    walkers: Any,
    weights: Array,
    e_estimate: Array,
    dt: float,
    n_meas: int,
) -> tuple[Array, Array, Array]:
    """Average n_meas trial-resampled mixed measurements with fixed walkers."""
    runtime_state = self._require_runtime_state(walkers)
    runtime_state, e_blk, sign_last, logov_last = self.measure_block_energy_state(
        hamil,
        walkers,
        weights,
        e_estimate,
        dt,
        n_meas,
        runtime_state,
    )
    self.import_runtime_state(runtime_state)
    self._bound_walkers = tree_map(jnp.asarray, walkers)
    return e_blk, sign_last, logov_last


def measure_block_energy_state(
    self,
    hamil: Any,
    walkers: Any,
    weights: Array,
    e_estimate: Array,
    dt: float,
    n_meas: int,
    runtime_state: RuntimeState,
) -> tuple[RuntimeState, Array, Array, Array]:
    walkers = tree_map(jnp.asarray, walkers)
    n_meas_i = max(int(n_meas), 1)
    steps = max(int(self.sample_update_steps), 0)
    chunk = int(getattr(self, "local_energy_chunk_size", 0))
    key = (id(hamil), steps, n_meas_i, chunk)
    fn = self._block_measure_fns.get(key)

    if fn is None:
        n_walkers = int(tree_leaves(walkers)[0].shape[0])
        chunk_i = chunk
        one_walker_eloc = lambda walker, bra_samples, mweights: jnp.einsum(
            "s,s->",
            mweights,
            jax.vmap(lambda bra: hamil.local_energy(bra, walker))(bra_samples),
        )

        def eval_e_loc_chunked(walkers_in, bra_in, mweights_in):
            if chunk_i <= 0 or chunk_i >= n_walkers:
                return jax.vmap(one_walker_eloc)(walkers_in, bra_in, mweights_in)
            e_chunks = []
            for start in range(0, n_walkers, chunk_i):
                end = min(start + chunk_i, n_walkers)
                e_chunks.append(
                    jax.vmap(one_walker_eloc)(
                        walkers_in[start:end],
                        bra_in[start:end],
                        mweights_in[start:end],
                    )
                )
            return jnp.concatenate(e_chunks, axis=0)

        def measure_fn(rng, st, fields, logsw, bra0, log_abs0, phase0, walkers_in, weights_in, e_est, dt_in):
            def advance_steps(rng_in, st_in, fields_in, logsw_in):
                if steps == 0:
                    return rng_in, st_in, fields_in, logsw_in

                sample_one = lambda k, walker, st_: self._pool_sampler.sample(k, (self.params, walker), st_)

                def step_body(_i, carry):
                    rng_c, st_c, f_c, lsw_c = carry
                    rng_c, sample_key = jax.random.split(rng_c)
                    keys = jax.random.split(sample_key, n_walkers)
                    st_c, (f_c, lsw_c) = jax.vmap(sample_one)(keys, walkers_in, st_c)
                    return (rng_c, st_c, f_c, lsw_c)

                return lax.fori_loop(
                    0,
                    steps,
                    step_body,
                    (rng_in, st_in, fields_in, logsw_in),
                )

            sign0, logov0, _ = self._calc_overlap_bundle_from_cache(
                walkers_in, bra0, log_abs0, phase0
            )
            clip = jnp.sqrt(2.0 / dt_in)

            def meas_body(_i, carry):
                rng_c, st_c, f_c, lsw_c, e_acc, sign_last, logov_last = carry
                rng_c, st_c, f_c, lsw_c = advance_steps(rng_c, st_c, f_c, lsw_c)

                bra_c, logcoef_c = self._evaluate_pool_bra(f_c, lsw_c)
                log_abs_c = jnp.real(logcoef_c)
                phase_c = jnp.exp(1.0j * jnp.imag(logcoef_c))
                sign_c, logov_c, mix_weights_c = self._calc_overlap_bundle_from_cache(
                    walkers_in, bra_c, log_abs_c, phase_c
                )

                e_loc = eval_e_loc_chunked(walkers_in, bra_c, mix_weights_c)
                e_loc = jnp.real(e_loc)
                e_loc = jnp.where(
                    jnp.abs(e_loc - e_est) > clip, e_est, e_loc
                )
                wsum = jnp.maximum(jnp.sum(weights_in), 1.0e-12)
                e_blk = jnp.sum(weights_in * e_loc) / wsum
                return (rng_c, st_c, f_c, lsw_c, e_acc + e_blk, sign_c, logov_c)

            return lax.fori_loop(
                0,
                n_meas_i,
                meas_body,
                (
                    rng,
                    st,
                    fields,
                    logsw,
                    jnp.array(0.0, dtype=jnp.float64),
                    sign0,
                    logov0,
                ),
            )

        fn = jax.jit(measure_fn)
        self._block_measure_fns[key] = fn

    rng_new, st_new, fields_new, logsw_new, e_acc, sign_last, logov_last = fn(
        runtime_state["rng"],
        runtime_state["pool_state"],
        runtime_state["pool_fields_tree"],
        runtime_state["pool_logsw"],
        runtime_state["pool_bra"],
        runtime_state["pool_log_abs"],
        runtime_state["pool_phase"],
        walkers,
        weights,
        e_estimate,
        jnp.asarray(dt),
    )
    bra_new, log_abs_new, phase_new = self._cache_from_samples(fields_new, logsw_new)
    runtime_state_new = self._pack_runtime_state(
        rng=rng_new,
        pool_state=st_new,
        pool_fields_tree=fields_new,
        pool_logsw=logsw_new,
        pool_bra=bra_new,
        pool_log_abs=log_abs_new,
        pool_phase=phase_new,
    )
    return runtime_state_new, e_acc / float(n_meas_i), sign_last, logov_last


__all__ = [
    "_cache_from_samples",
    "_overlap_bundle_from_state",
    "_pack_runtime_state",
    "_require_runtime_state",
    "bind_walkers",
    "calc_force_bias",
    "calc_force_bias_state",
    "calc_local_energy",
    "calc_local_energy_state",
    "calc_slov",
    "calc_slov_state",
    "export_runtime_state",
    "get_rdm1",
    "import_runtime_state",
    "init_runtime_state",
    "init_walkers",
    "on_walkers_updated",
    "on_walkers_updated_state",
    "orthonormalize_walkers",
    "measure_block_energy",
    "measure_block_energy_state",
    "stochastic_reconfiguration",
    "stochastic_reconfiguration_state",
    "update_tethered_samples",
    "update_tethered_samples_state",
]
