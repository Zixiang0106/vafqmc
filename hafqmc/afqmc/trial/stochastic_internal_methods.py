"""Internal helper methods for stochastic VAFQMC trial."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging

import jax
from jax import lax, numpy as jnp
from jax.tree_util import tree_leaves, tree_map

from ...hamiltonian import calc_rdm, calc_slov
from ...propagator import orthonormalize
from ...sampler import choose_sampler_maker, make_batched
from ..utils import (
    PMAP_AXIS_NAME,
    _spin_sum_rdm,
    gaussian_logdens,
    mix_overlap_terms as _mix_overlap_terms,
)
from ..walker import AFQMCState, init_walkers

Array = jnp.ndarray


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


def _ensure_pool_field_batch_size(self, fields_tree: Any, n_samples: int) -> Any:
    """Resize walker-local pool fields to exactly n_samples along axis=1."""
    cur = int(tree_leaves(fields_tree)[0].shape[1])
    if cur == n_samples:
        return fields_tree
    if cur > n_samples:
        return tree_map(lambda x: x[:, :n_samples], fields_tree)

    reps = (n_samples + cur - 1) // cur

    def _tile_to(x):
        tail = (1,) * max(x.ndim - 2, 0)
        xt = jnp.tile(x, (1, reps) + tail)
        return xt[:, :n_samples]

    return tree_map(_tile_to, fields_tree)


def _replace_sampler_state_fields(state: Any, fields: Any) -> Any:
    """Replace the position/field part of sampler state for tuple-style samplers."""
    if isinstance(state, tuple):
        if len(state) == 0:
            return state
        return (fields,) + tuple(state[1:])
    if isinstance(state, list):
        if len(state) == 0:
            return state
        out = list(state)
        out[0] = fields
        return out
    raise TypeError(
        "Unsupported sampler state type for field override: "
        f"{type(state).__name__}. Tuple/list states are supported."
    )


def _get_init_walkers_infer_fn(self):
    fn = getattr(self, "_init_walkers_infer_fn", None)
    if fn is not None:
        return fn

    hamil = getattr(self, "_hamiltonian", None)
    if hamil is None:
        return None

    apply_ansatz = jax.vmap(lambda f: self.ansatz.apply(self.params, f))
    eval_eloc = jax.vmap(lambda b, k: hamil.local_energy(b, k))
    eval_slov = jax.vmap(lambda b, k: calc_slov(b, k))
    floor = float(getattr(self, "logdens_floor", -60.0))
    local_chunk = int(getattr(self, "local_energy_chunk_size", 0))
    # Stage-1 inference has no pool-sample axis. A practical auto rule is to scale
    # AFQMC local-energy chunk by n_samples to keep similar effective workload.
    chunk = int(local_chunk * int(getattr(self, "n_samples", 1))) if local_chunk > 0 else 0

    def eval_fn(fields_pair: Any, logsw: Array):
        bra_fields = tree_map(lambda x: x[:, 0], fields_pair)
        ket_fields = tree_map(lambda x: x[:, 1], fields_pair)
        bra, bra_lw = apply_ansatz(bra_fields)
        ket, ket_lw = apply_ansatz(ket_fields)
        n_walkers = int(tree_leaves(ket)[0].shape[0])
        if chunk <= 0 or chunk >= n_walkers:
            eloc = eval_eloc(bra, ket)
            sign, logabs = eval_slov(bra, ket)
        else:
            eloc_chunks = []
            sign_chunks = []
            logabs_chunks = []
            for start in range(0, n_walkers, chunk):
                end = min(start + chunk, n_walkers)
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

        dlog = jnp.real(logov - logsw)
        dlog = jnp.where(jnp.isfinite(dlog), dlog, floor)
        dmax = jnp.max(dlog)
        rel_w = jnp.exp(dlog - dmax)
        rel_w = rel_w / jnp.maximum(jnp.mean(rel_w), 1.0e-12)

        exp_es = jnp.mean((eloc * sign) * rel_w)
        exp_s = jnp.mean(sign * rel_w)
        denom = jnp.real(exp_s)
        etot = jnp.where(jnp.abs(denom) > 1.0e-12, jnp.real(exp_es) / denom, jnp.nan)
        return etot, jnp.real(exp_es), jnp.real(exp_s)

    # For chunked inference, keep eager mode to avoid giant unrolled HLO from
    # Python chunk loops and reduce compilation memory pressure.
    fn = eval_fn if chunk > 0 else jax.jit(eval_fn)
    self._init_walkers_infer_fn = fn
    return fn


def _sample_walkers_from_trial(self, n_walkers: int, key: Array) -> Any:
    """Initialize walkers by sampling ansatz outputs from trial distribution."""
    logger = logging.getLogger("hafqmc.afqmc")
    logger.info(
        "Initializing walkers from trial sampler: n_walkers=%d sampler=%s burn_in=%d",
        int(n_walkers),
        str(getattr(self, "_sampler_name", "unknown")),
        int(self.init_walkers_burn_in),
    )
    chains_per_walker = int(getattr(self, "init_walkers_chains_per_walker", 0))
    if chains_per_walker <= 0:
        chains_per_walker = int(self.n_samples)
    n_init_chains = int(n_walkers * chains_per_walker)
    logger.info(
        "Trial & walker burn-in: total_chains=%d (%d walkers x %d chains/walker)",
        n_init_chains,
        int(n_walkers),
        int(chains_per_walker),
    )

    # Fixed init-walker target: mimic train.py burn-in
    # log p(field_pair) ~ Re[ log <Psi(field_bra)|Psi(field_ket)> + lw_bra + lw_ket ].
    pair_fields_shape = tree_map(
        lambda s: jnp.asarray((2, *tuple(map(int, jnp.asarray(s).reshape(-1).tolist())))),
        self.fields_shape,
    )

    def pair_logdens(params: Any, fields_pair: Any) -> Array:
        bra_fields = tree_map(lambda x: x[0], fields_pair)
        ket_fields = tree_map(lambda x: x[1], fields_pair)
        bra, bra_lw = self.ansatz.apply(params, bra_fields)
        ket, ket_lw = self.ansatz.apply(params, ket_fields)
        _sign, logov = calc_slov(bra, ket)
        logd = jnp.real(logov + bra_lw + ket_lw)
        return jnp.where(jnp.isfinite(logd), logd, self.logdens_floor)

    init_sampler = choose_sampler_maker(self._sampler_name)(
        pair_logdens,
        pair_fields_shape,
        **self._sampler_kwargs,
    )
    sampler_params = self.params

    walker_sampler = make_batched(init_sampler, n_init_chains, concat=False)

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

    infer_steps = int(getattr(self, "init_walkers_infer_steps", 0))
    if infer_steps > 0:
        infer_fn = _get_init_walkers_infer_fn(self)
        if infer_fn is None:
            logger.warning(
                "Trial inference skipped: no hamiltonian bound in trial object."
            )
        else:
            logger.info(
                "Starting trial energy inference for %d steps",
                int(infer_steps),
            )
            infer_state = state
            for ii in range(infer_steps):
                key, infer_key = jax.random.split(key)
                infer_state, (infer_fields, infer_logsw) = walker_sampler.sample(
                    infer_key, sampler_params, infer_state
                )
                e_infer, exp_es, exp_s = infer_fn(infer_fields, infer_logsw)
                logger.info(
                    "Infer %d/%d e_vafqmc=%.12f exp_es=%.12f exp_s=%.12f",
                    ii + 1,
                    int(infer_steps),
                    float(e_infer),
                    float(exp_es),
                    float(exp_s),
                )
    key, sample_key = jax.random.split(key)
    _, (fields, _logsw) = walker_sampler.sample(sample_key, sampler_params, state)
    fields_for_walkers = tree_map(lambda x: x[:, 1], fields)
    walkers_all, _ = jax.vmap(lambda f: self.ansatz.apply(self.params, f))(fields_for_walkers)
    left_fields_all = tree_map(lambda x: x[:, 0], fields)

    if chains_per_walker > 1:
        key, pick_key = jax.random.split(key)
        pick = jax.random.randint(pick_key, (n_walkers,), 0, chains_per_walker)
        idx = jnp.arange(n_walkers)

        def _pick_one(x):
            x = x.reshape((n_walkers, chains_per_walker) + x.shape[1:])
            return x[idx, pick]

        walkers = tree_map(_pick_one, walkers_all)

        # keep a walker-local pool of left fields and force sample-0 to be
        # the one corresponding to the chosen right-side walker.
        def _group(x):
            return x.reshape((n_walkers, chains_per_walker) + x.shape[1:])

        grouped_left = tree_map(_group, left_fields_all)
        pool_seed = _ensure_pool_field_batch_size(self, grouped_left, int(self.n_samples))
        left_selected = tree_map(_pick_one, left_fields_all)
        pool_seed = tree_map(lambda x, s: x.at[:, 0].set(s), pool_seed, left_selected)
        self._init_pool_fields_override = pool_seed
    else:
        walkers = walkers_all
        left_selected = left_fields_all
        pool_seed = tree_map(
            lambda x: jnp.repeat(x[:, None], int(self.n_samples), axis=1),
            left_selected,
        )
        self._init_pool_fields_override = pool_seed
    logger.info(
        "End of walker burn-in: total walkers=%d.",
        int(n_walkers),
    )
    if isinstance(walkers, (tuple, list)) and len(walkers) == 2:
        walkers = (walkers[0], walkers[1])
    return walkers


def _clone_stochastic_trial_for_runtime_init(self, *, seed: int):
    from .stochastic import VAFQMCTrial

    return VAFQMCTrial(
        self.ansatz,
        self.params,
        hamiltonian=getattr(self, "_hamiltonian", None),
        reference_wfn=self.reference_wfn,
        n_samples=int(self.n_samples),
        burn_in=int(self.burn_in),
        sampler_name=str(self._sampler_name),
        sampler_kwargs=dict(getattr(self, "_sampler_kwargs", {}) or {}),
        sampling_target=str(self.sampling_target),
        logdens_floor=float(self.logdens_floor),
        sample_update_steps=int(self.sample_update_steps),
        local_energy_chunk_size=int(self.local_energy_chunk_size),
        init_walkers_from_trial=bool(self.init_walkers_from_trial),
        init_walkers_burn_in=int(self.init_walkers_burn_in),
        init_walkers_chains_per_walker=int(self.init_walkers_chains_per_walker),
        init_walkers_infer_steps=int(self.init_walkers_infer_steps),
        max_prop=self.max_prop,
        seed=int(seed),
    )


def _make_stochastic_init_infer_pm(self):
    hamil = getattr(self, "_hamiltonian", None)
    if hamil is None:
        return None

    apply_ansatz = jax.vmap(lambda f: self.ansatz.apply(self.params, f))
    eval_eloc = jax.vmap(lambda b, k: hamil.local_energy(b, k))
    eval_slov = jax.vmap(lambda b, k: calc_slov(b, k))
    floor = float(getattr(self, "logdens_floor", -60.0))
    local_chunk = int(getattr(self, "local_energy_chunk_size", 0))
    chunk = int(local_chunk * int(getattr(self, "n_samples", 1))) if local_chunk > 0 else 0

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


def _build_stochastic_init_context_local(self, n_local: int, key: Any, device: Any):
    chains_per_walker = int(getattr(self, "init_walkers_chains_per_walker", 0))
    if chains_per_walker <= 0:
        chains_per_walker = int(self.n_samples)
    n_init_chains = int(n_local * chains_per_walker)

    pair_fields_shape = tree_map(
        lambda s: jnp.asarray((2, *tuple(map(int, jnp.asarray(s).reshape(-1).tolist())))),
        self.fields_shape,
    )

    def pair_logdens(params: Any, fields_pair: Any) -> Array:
        bra_fields = tree_map(lambda x: x[0], fields_pair)
        ket_fields = tree_map(lambda x: x[1], fields_pair)
        bra, bra_lw = self.ansatz.apply(params, bra_fields)
        ket, ket_lw = self.ansatz.apply(params, ket_fields)
        _sign, logov = calc_slov(bra, ket)
        logd = jnp.real(logov + bra_lw + ket_lw)
        return jnp.where(jnp.isfinite(logd), logd, self.logdens_floor)

    init_sampler = choose_sampler_maker(str(self._sampler_name))(
        pair_logdens,
        pair_fields_shape,
        **dict(getattr(self, "_sampler_kwargs", {}) or {}),
    )
    walker_sampler = make_batched(init_sampler, n_init_chains, concat=False)
    sampler_params = self.params

    with jax.default_device(device):
        stage_key, init_key = jax.random.split(key)
        sample_state = walker_sampler.init(init_key, sampler_params)
        if int(self.init_walkers_burn_in) > 0:
            stage_key, burn_key = jax.random.split(stage_key)
            sample_state = walker_sampler.burn_in(
                burn_key,
                sampler_params,
                sample_state,
                int(self.init_walkers_burn_in),
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


def _sample_stochastic_infer_local(self, ctx: _StochasticInitContext):
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
    self,
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
        walkers_all, _ = jax.vmap(lambda f: self.ansatz.apply(self.params, f))(fields_for_walkers)
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
            pool_seed = _ensure_pool_field_batch_size(self, grouped_left, int(self.n_samples))
            left_selected = tree_map(_pick_one, left_fields_all)
            pool_seed = tree_map(lambda x, s: x.at[:, 0].set(s), pool_seed, left_selected)
        else:
            walkers = walkers_all
            left_selected = left_fields_all
            pool_seed = tree_map(
                lambda x: jnp.repeat(x[:, None], int(self.n_samples), axis=1),
                left_selected,
            )

        if isinstance(walkers, (tuple, list)) and len(walkers) == 2:
            w_up, w_dn = walkers
            walkers = (w_up.astype(jnp.complex128), w_dn.astype(jnp.complex128))
        else:
            walkers = walkers.astype(jnp.complex128)

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
    self,
    hamil: Any,
    walkers: Any,
    pool_seed: Any,
    key: Any,
    device: Any,
    e_est_init: Any,
) -> tuple[AFQMCState, float]:
    trial_local = self._clone_stochastic_trial_for_runtime_init(
        seed=int(jax.device_get(key)[0])
    )
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
            e_local = jnp.sum(
                jnp.real(trial_local.calc_local_energy_state(hamil, walkers, trial_state))
            )
            e_sum = float(jax.device_get(e_local))

        state = AFQMCState(
            walkers=walkers,
            weights=weights,
            sign=sign,
            logov=logov,
            key=key_out,
            e_estimate=jnp.asarray(0.0, dtype=jnp.float64),
            pop_control_shift=jnp.asarray(0.0, dtype=jnp.float64),
            walker_fields=(
                trial_state["walker_fields"]
                if isinstance(trial_state, dict) and "walker_fields" in trial_state
                else (
                    trial_local.walker_fields
                    if getattr(trial_local, "walker_fields", None) is not None
                    else jnp.zeros((0, 0, 0), dtype=jnp.float64)
                )
            ),
            trial_state=trial_state,
        )
    return state, e_sum


def _initialize_stochastic_reference_walkers_local(
    self,
    n_local: int,
    key: Any,
    noise: float,
    device: Any,
) -> Any:
    trial_local = self._clone_stochastic_trial_for_runtime_init(
        seed=int(jax.device_get(key)[0])
    )
    with jax.default_device(device):
        walkers, _ = init_walkers(trial_local, n_local, key, noise=noise)
        if isinstance(walkers, (tuple, list)) and len(walkers) == 2:
            w_up, w_dn = walkers
            walkers = (w_up.astype(jnp.complex128), w_dn.astype(jnp.complex128))
        else:
            walkers = walkers.astype(jnp.complex128)
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


def _init_pool_state(self, walkers: Any, burn_steps: int, init_fields: Any = None):
    self._rng, init_key = jax.random.split(self._rng)
    init_keys = jax.random.split(init_key, int(tree_leaves(walkers)[0].shape[0]))
    state = jax.vmap(lambda k, w: self._pool_sampler.init(k, (self.params, w)))(init_keys, walkers)

    if init_fields is not None:
        # Inject provided fields then refresh log-density/grad caches
        # under walker-conditioned runtime target.
        state_base = state
        try:
            init_fields_flat = self._fields_tree_to_flat(init_fields)
            state_trial = _replace_sampler_state_fields(state, init_fields_flat)
            state_trial = jax.vmap(lambda st, w: self._pool_sampler.refresh(st, (self.params, w)))(
                state_trial,
                walkers,
            )
            state = state_trial
        except Exception as exc: 
            state = state_base
            logging.getLogger("hafqmc.afqmc").warning(
                "Stage-1 handoff fields could not be applied; fallback to default pool init. "
                "error=%s",
                exc,
            )

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
    "_ensure_pool_field_batch_size",
    "_get_force_bias_fn",
    "_get_local_energy_fn",
    "_init_pool_state",
    "_replace_sampler_state_fields",
    "_sample_pool_step",
    "_sample_walkers_from_trial",
    "_update_pool_cache",
]
