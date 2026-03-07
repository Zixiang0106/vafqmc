"""Internal helper methods for stochastic VAFQMC trial."""

from __future__ import annotations

from typing import Any
import logging

import jax
from jax import numpy as jnp
from jax.tree_util import tree_leaves, tree_map

from ...hamiltonian import _has_spin, calc_rdm, calc_slov
from ...sampler import choose_sampler_maker, make_batched
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


def _get_init_walkers_diag_fn(self):
    fn = getattr(self, "_init_walkers_diag_fn", None)
    if fn is not None:
        return fn

    hamil = getattr(self, "_hamiltonian", None)
    if hamil is None:
        return None

    apply_ansatz = jax.vmap(lambda f: self.ansatz.apply(self.params, f))
    eval_eloc = jax.vmap(lambda b, k: hamil.local_energy(b, k))
    eval_slov = jax.vmap(lambda b, k: calc_slov(b, k))
    floor = float(getattr(self, "logdens_floor", -60.0))

    def eval_fn(fields_pair: Any, logsw: Array):
        bra_fields = tree_map(lambda x: x[:, 0], fields_pair)
        ket_fields = tree_map(lambda x: x[:, 1], fields_pair)
        bra, bra_lw = apply_ansatz(bra_fields)
        ket, ket_lw = apply_ansatz(ket_fields)
        eloc = eval_eloc(bra, ket)
        sign, logabs = eval_slov(bra, ket)
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

    fn = jax.jit(eval_fn)
    self._init_walkers_diag_fn = fn
    return fn


def _project_ghf_walkers_to_uhf(self, walkers_ghf: Array) -> Any:
    """Project sampled GHF walkers into spin-separated (w_up, w_dn) walkers.

    This is an initialization-time approximation used when the trial ansatz
    outputs a spin-mixed/GHF determinant but AFQMC walkers are spin separated.
    """
    if not _has_spin(self.reference_wfn):
        raise ValueError(
            "Cannot project GHF sampled walkers to UHF: reference_wfn is not spin-separated."
        )

    nbasis = int(self.nbasis)
    nrow = int(walkers_ghf.shape[1])
    if nrow != 2 * nbasis:
        raise ValueError(
            f"Unexpected GHF walker row dimension: got {nrow}, expected {2 * nbasis}."
        )

    n_alpha = int(self.reference_wfn[0].shape[1])
    n_beta = int(self.reference_wfn[1].shape[1])
    ncol = int(walkers_ghf.shape[2])
    if n_alpha > ncol or n_beta > ncol:
        raise ValueError(
            f"Cannot project GHF walkers: ncol={ncol}, n_alpha={n_alpha}, n_beta={n_beta}."
        )

    def _proj_occ(block: Array, nocc: int) -> Array:
        # Use principal left-singular subspace (less sensitive than QR
        # to column ordering for spin-mixed/GHF walkers).
        u, _s, _vh = jnp.linalg.svd(block, full_matrices=False)
        return u[:, :nocc]

    def _proj_one(w: Array) -> Any:
        up_block = w[:nbasis, :]
        dn_block = w[nbasis:, :]
        return _proj_occ(up_block, n_alpha), _proj_occ(dn_block, n_beta)

    return jax.vmap(_proj_one)(walkers_ghf)


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
        "Stage-1 trial burn-in: total_chains=%d (%d walkers x %d chains/walker)",
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

    def train_style_logdens(params: Any, fields_pair: Any) -> Array:
        bra_fields = tree_map(lambda x: x[0], fields_pair)
        ket_fields = tree_map(lambda x: x[1], fields_pair)
        bra, bra_lw = self.ansatz.apply(params, bra_fields)
        ket, ket_lw = self.ansatz.apply(params, ket_fields)
        _sign, logov = calc_slov(bra, ket)
        logd = jnp.real(logov + bra_lw + ket_lw)
        return jnp.where(jnp.isfinite(logd), logd, self.logdens_floor)

    init_sampler = choose_sampler_maker(self._sampler_name)(
        train_style_logdens,
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

    diag_steps = int(getattr(self, "init_walkers_diag_steps", 0))
    if diag_steps > 0:
        diag_fn = _get_init_walkers_diag_fn(self)
        if diag_fn is None:
            logger.warning(
                "Stage-1 trial diagnostics skipped: no hamiltonian bound in trial object."
            )
        else:
            logger.info(
                "Stage-1 trial diagnostics (VAFQMC inference): steps=%d",
                int(diag_steps),
            )
            diag_state = state
            for ii in range(diag_steps):
                key, diag_key = jax.random.split(key)
                diag_state, (diag_fields, diag_logsw) = walker_sampler.sample(
                    diag_key, sampler_params, diag_state
                )
                e_diag, exp_es, exp_s = diag_fn(diag_fields, diag_logsw)
                logger.info(
                    "Stage-1 infer %d/%d e_vafqmc=%.12f exp_es=%.12f exp_s=%.12f",
                    ii + 1,
                    int(diag_steps),
                    float(e_diag),
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

        # Stage-1 -> Stage-2 handoff:
        # keep a walker-local pool of left fields and force sample-0 to be
        # the one corresponding to the chosen right-side walker.
        def _group(x):
            return x.reshape((n_walkers, chains_per_walker) + x.shape[1:])

        grouped_left = tree_map(_group, left_fields_all)
        pool_seed = _ensure_pool_field_batch_size(self, grouped_left, int(self.n_samples))
        left_selected = tree_map(_pick_one, left_fields_all)
        pool_seed = tree_map(lambda x, s: x.at[:, 0].set(s), pool_seed, left_selected)
        self._init_pool_fields_override = pool_seed

        logger.info(
            "Stage-1 trial burn-in: selected 1 ket per walker from %d candidates.",
            int(chains_per_walker),
        )
        logger.info(
            "Stage-1 handoff prepared: left pool initialized from corresponding chain fields "
            "(n_samples=%d).",
            int(self.n_samples),
        )
    else:
        walkers = walkers_all
        left_selected = left_fields_all
        pool_seed = tree_map(
            lambda x: jnp.repeat(x[:, None], int(self.n_samples), axis=1),
            left_selected,
        )
        self._init_pool_fields_override = pool_seed
        logger.info(
            "Stage-1 handoff prepared from single-chain fields (replicated to n_samples=%d).",
            int(self.n_samples),
        )
    if isinstance(walkers, (tuple, list)) and len(walkers) == 2:
        walkers = (walkers[0], walkers[1])
        if not _has_spin(walkers):
            raise ValueError("init_walkers_from_trial requires spin-separated ansatz output.")
        logger.info("Initial walkers sampled from trial successfully (spin-separated).")
        return walkers

    # Spin-mixed/GHF ansatz output: optionally keep as GHF or project to spin-separated.
    proj_mode = str(getattr(self, "init_walkers_projection", "auto")).lower()
    if proj_mode in ("keep", "none", "ghf", "no_projection"):
        walkers_arr = jnp.asarray(walkers)
        if walkers_arr.ndim != 3:
            raise ValueError(
                f"Unsupported sampled GHF walker shape for keep mode: {walkers_arr.shape}."
            )
        logger.info(
            "Initial walkers sampled from trial as GHF and kept in GHF form "
            "(mode=%s, shape=%s).",
            proj_mode,
            tuple(map(int, walkers_arr.shape)),
        )
        return walkers_arr

    if proj_mode in ("auto", "ghf_to_uhf", "project", "drop_spin_mixing"):
        walkers_arr = jnp.asarray(walkers)
        if walkers_arr.ndim != 3:
            raise ValueError(
                f"Unsupported sampled walker shape for projection: {walkers_arr.shape}."
            )
        walkers = _project_ghf_walkers_to_uhf(self, walkers_arr)
        logger.info(
            "Initial walkers sampled from trial as GHF and projected to spin-separated form "
            "(mode=%s, shape=%s).",
            proj_mode,
            tuple(map(int, walkers_arr.shape)),
        )
        return walkers

    raise ValueError(
        "init_walkers_from_trial got non-spin-separated ansatz output (likely spin-mixed/GHF), "
        f"and init_walkers_projection={proj_mode!r} forbids projection. "
        "Set cfg.stochastic_trial.init_walkers_projection='auto' (or 'ghf_to_uhf'), "
        "or use 'keep' to propagate GHF walkers."
    )


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
            # Sampler internals store flattened coordinates as first state slot.
            init_fields_flat = self._fields_tree_to_flat(init_fields)
            state_trial = _replace_sampler_state_fields(state, init_fields_flat)
            state_trial = jax.vmap(lambda st, w: self._pool_sampler.refresh(st, (self.params, w)))(
                state_trial,
                walkers,
            )
            state = state_trial
        except Exception as exc:  # pragma: no cover - defensive fallback
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
    "_project_ghf_walkers_to_uhf",
    "_replace_sampler_state_fields",
    "_sample_pool_step",
    "_sample_walkers_from_trial",
    "_update_pool_cache",
]
