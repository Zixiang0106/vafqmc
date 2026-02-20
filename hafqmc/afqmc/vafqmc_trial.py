"""VAFQMC trial adapter with tethered (per-walker) sampling.

This module implements a composite-walker view:
- AFQMC walkers evolve in determinant space.
- Each walker owns an auxiliary-field pool of size ``P``.
- The pool is sampled from a log-density conditioned on the current walker.
- After each walker propagation step, the pool is advanced by ``M`` sampler
  steps and an extra hand-off phaseless weight is applied.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import jax
import numpy as onp
from jax import lax, numpy as jnp
from jax.tree_util import tree_leaves, tree_map

from ..ansatz import Ansatz
from ..hamiltonian import _has_spin, calc_rdm, calc_slov
from ..propagator import orthonormalize
from ..sampler import choose_sampler_maker, make_batched
from ..utils import load_pickle
from .afqmc_utils import _spin_sum_rdm

Array = jnp.ndarray


def _gaussian_logdens(_params: Any, fields: Any) -> Array:
    """Unnormalized log-density of N(0, I) over an arbitrary pytree."""
    vals = [jnp.real(x * jnp.conj(x)).sum() for x in tree_leaves(fields)]
    return -0.5 * jnp.sum(jnp.asarray(vals))


def _extract_params_from_payload(payload: Any) -> Mapping[str, Any]:
    """Best-effort extraction of flax params from training/checkpoint payload."""
    if isinstance(payload, Mapping) and "params" in payload:
        return payload

    # Common training checkpoint layout:
    # (rng_key, (step, params, mc_state, opt_state, ...))
    if isinstance(payload, (tuple, list)) and len(payload) == 2:
        maybe_state = payload[1]
        if isinstance(maybe_state, (tuple, list)) and len(maybe_state) >= 2:
            maybe_params = maybe_state[1]
            if isinstance(maybe_params, Mapping) and "params" in maybe_params:
                return maybe_params

    if isinstance(payload, Mapping):
        for v in payload.values():
            try:
                return _extract_params_from_payload(v)
            except ValueError:
                pass
    if isinstance(payload, (tuple, list)):
        for v in payload:
            try:
                return _extract_params_from_payload(v)
            except ValueError:
                pass

    raise ValueError("Cannot locate flax params in checkpoint payload.")


def _extract_params(payload: Any) -> Mapping[str, Any]:
    """Match the tuple-unpack logic used in ``hafqmc.train``."""
    params = payload
    if isinstance(params, tuple):
        params = params[1]
    if isinstance(params, tuple):
        params = params[1]
    if isinstance(params, Mapping) and "params" in params:
        return params
    raise ValueError("Payload does not match train.py checkpoint tuple layout.")


def _normalize_ansatz_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Convert braket checkpoint params to ansatz.apply params if needed."""
    if not isinstance(params, Mapping) or "params" not in params:
        return params
    inner = params["params"]
    if isinstance(inner, Mapping) and "ansatz" in inner:
        return {"params": inner["ansatz"]}
    return params


def _mix_overlap_terms(
    sample_sign: Array,
    sample_logov: Array,
    sample_log_abs: Array,
    sample_phase: Array,
) -> tuple[Array, Array, Array]:
    """Combine per-sample overlaps into total sign/logov and normalized weights."""
    log_terms = sample_logov + sample_log_abs
    phases = sample_sign * sample_phase

    max_log = jnp.max(jnp.real(log_terms))
    scaled = phases * jnp.exp(log_terms - max_log)
    total = jnp.sum(scaled)
    total_abs = jnp.abs(total)
    valid = total_abs > 0

    sign = jnp.where(valid, total / total_abs, 0.0 + 0.0j)
    logov = jnp.where(valid, max_log + jnp.log(total_abs), -jnp.inf)
    weights = jnp.where(valid, scaled / total, jnp.zeros_like(scaled))
    return sign, logov, weights


def _phaseless_from_ratio(ratio: Array) -> Array:
    theta = jnp.angle(ratio)
    handoff = jnp.abs(ratio) * jnp.cos(theta)
    handoff = jnp.real(handoff)
    handoff = jnp.where(jnp.isnan(handoff), 0.0, handoff)
    handoff = jnp.where(handoff < 0.0, 0.0, handoff)
    return handoff


class VAFQMCTrial:
    """Custom trial object consumed by ``hafqmc.afqmc.afqmc_energy``.

    Tethered mode:
    - ``walker_fields`` stores per-walker field pools with shape
      ``(n_walkers, n_samples, n_fields_flat)``.
    - sampler target depends on each walker, not only the fixed reference.
    """

    def __init__(
        self,
        ansatz: Ansatz,
        params: Mapping[str, Any],
        *,
        reference_wfn: Any,
        n_samples: int = 128,
        burn_in: int = 256,
        sampler_name: str = "hmc",
        sampler_kwargs: Optional[Mapping[str, Any]] = None,
        sampling_target: str = "walker_overlap",
        logdens_floor: float = -60.0,
        refresh_interval: int = 0,
        replace_per_refresh: int = 1,
        sample_update_steps: int = 1,
        pure_eval_pairs: int = 1000,
        max_prop: Optional[Any] = None,
        seed: int = 0,
    ) -> None:
        self.ansatz = ansatz
        self.params = params
        self.reference_wfn = tree_map(jnp.asarray, reference_wfn)
        if _has_spin(self.reference_wfn):
            self.nbasis = int(self.reference_wfn[0].shape[0])
        else:
            self.nbasis = int(self.reference_wfn.shape[0])

        self.n_samples = int(n_samples)
        self.burn_in = int(burn_in)
        self.sampling_target = str(sampling_target).lower()
        self.logdens_floor = float(logdens_floor)
        self.refresh_interval = int(refresh_interval)
        self.replace_per_refresh = int(replace_per_refresh)
        self.sample_update_steps = int(sample_update_steps)
        self.pure_eval_pairs = int(pure_eval_pairs)
        self.max_prop = max_prop
        self.seed = int(seed)
        if self.n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")

        self.fields_shape = self.ansatz.fields_shape(self.max_prop)
        self._zero_fields = tree_map(
            lambda s: jnp.zeros(tuple(onp.asarray(s).tolist())),
            self.fields_shape,
        )
        self.n_fields = int(sum(int(onp.prod(x.shape)) for x in tree_leaves(self._zero_fields)))

        name = sampler_name.lower()
        kwargs = dict(sampler_kwargs or {})
        if name in ("hmc", "hamiltonian", "hybrid") and not kwargs:
            kwargs = {"dt": 0.1, "length": 1.0}
        self._sampler_name = name
        self._sampler_kwargs = kwargs
        self._sampler = choose_sampler_maker(name)(self._build_logdens(), self.fields_shape, **kwargs)
        self._sampler_batched = make_batched(self._sampler, self.n_samples, concat=False)

        self._walker_rng = jax.random.PRNGKey(self.seed)
        self._pair_rng = jax.random.PRNGKey(self.seed + 7919)

        # Pair sampler for inference-like pure estimator, lazily initialized.
        self._pair_fields_shape = tree_map(
            lambda s: onp.array((2, *tuple(onp.asarray(s).tolist()))),
            self.fields_shape,
        )
        self._pair_sampler = None
        self._pair_state = None

        self._n_walkers = 0
        self._current_walkers = None
        self._walker_sampler_state = None

        # Tethered pool/cache (per walker).
        self.walker_fields = None  # (n_walkers, n_samples, n_fields_flat)
        self.walker_fields_tree = None  # pytree with leading (n_walkers, n_samples)
        self.walker_logsw = None  # (n_walkers, n_samples)
        self.walker_log_abs = None  # (n_walkers, n_samples)
        self.walker_phase = None  # (n_walkers, n_samples)
        self.walker_bra_samples = None  # pytree with leading (n_walkers, n_samples)

        # Compatibility aliases used by debugging estimators.
        self.sample_fields = None
        self.sample_logsw = None
        self.sample_log_abs = None
        self.sample_phase = None
        self.bra_samples = None

        # Use deterministic reference RDM for AFQMC MF-shift setup.
        self.rdm1 = jnp.real(self._collapse_rdm(calc_rdm(self.reference_wfn, self.reference_wfn)))

    def _build_logdens(self):
        if self.sampling_target in ("gaussian", "normal", "std"):
            return _gaussian_logdens

        # Backward-compatible fixed-reference target.
        if self.sampling_target in ("coeff_overlap", "reference_overlap"):
            def _coeff_overlap_logdens(params: Any, fields: Any) -> Array:
                ansatz_params = params[0] if isinstance(params, (tuple, list)) else params
                bra, bra_lw = self.ansatz.apply(ansatz_params, fields)
                _, logov_ref = calc_slov(bra, self.reference_wfn)
                logd = jnp.real(logov_ref + bra_lw)
                return jnp.where(jnp.isfinite(logd), logd, self.logdens_floor)

            return _coeff_overlap_logdens

        # Tethered target: p(x|phi_k) proportional to |<Psi_T(x)|phi_k>|.
        def _walker_overlap_logdens(params: Any, fields: Any) -> Array:
            if isinstance(params, (tuple, list)) and len(params) == 2:
                ansatz_params, walker = params
            else:
                ansatz_params, walker = params, self.reference_wfn
            bra, bra_lw = self.ansatz.apply(ansatz_params, fields)
            _, logov = calc_slov(bra, walker)
            logd = jnp.real(logov + bra_lw)
            return jnp.where(jnp.isfinite(logd), logd, self.logdens_floor)

        return _walker_overlap_logdens

    @classmethod
    def from_checkpoint(
        cls,
        hamiltonian: Any,
        checkpoint_path: str,
        *,
        ansatz_cfg: Mapping[str, Any],
        n_samples: int = 20,
        burn_in: int = 1000,
        sampler_name: str = "hmc",
        sampler_kwargs: Optional[Mapping[str, Any]] = None,
        sampling_target: str = "walker_overlap",
        logdens_floor: float = -60.0,
        refresh_interval: int = 0,
        replace_per_refresh: int = 1,
        sample_update_steps: int = 1,
        pure_eval_pairs: int = 1000,
        max_prop: Optional[Any] = None,
        seed: int = 0,
    ) -> "VAFQMCTrial":
        payload = load_pickle(checkpoint_path)
        params = _extract_params(payload)
        params = _normalize_ansatz_params(params)
        ansatz = Ansatz.create(hamiltonian, **dict(ansatz_cfg))
        return cls(
            ansatz,
            params,
            reference_wfn=hamiltonian.wfn0,
            n_samples=n_samples,
            burn_in=burn_in,
            sampler_name=sampler_name,
            sampler_kwargs=sampler_kwargs,
            sampling_target=sampling_target,
            logdens_floor=logdens_floor,
            refresh_interval=refresh_interval,
            replace_per_refresh=replace_per_refresh,
            sample_update_steps=sample_update_steps,
            pure_eval_pairs=pure_eval_pairs,
            max_prop=max_prop,
            seed=seed,
        )

    def _collapse_rdm(self, rdm: Array) -> Array:
        """Convert spin-resolved/GHF RDM into spin-summed spatial RDM."""
        if rdm.ndim == 2 and rdm.shape[-1] == 2 * self.nbasis:
            blk = rdm.reshape(2, self.nbasis, 2, self.nbasis).swapaxes(1, 2)
            return blk[0, 0] + blk[1, 1]
        return _spin_sum_rdm(rdm)

    def get_rdm1(self) -> Array:
        return self.rdm1

    def init_walkers(self, n_walkers: int, key: Array, noise: float = 0.0) -> Any:
        if not _has_spin(self.reference_wfn):
            raise ValueError("VAFQMCTrial requires spin-separated reference_wfn for AFQMC walkers.")

        w_up, w_dn = self.reference_wfn
        w_up = jnp.broadcast_to(w_up, (n_walkers,) + w_up.shape)
        w_dn = jnp.broadcast_to(w_dn, (n_walkers,) + w_dn.shape)

        if noise > 0.0:
            key, k1, k2 = jax.random.split(key, 3)
            w_up = w_up + noise * jax.random.normal(k1, w_up.shape)
            w_dn = w_dn + noise * jax.random.normal(k2, w_dn.shape)

        walkers, _ = orthonormalize((w_up, w_dn))
        return walkers

    def orthonormalize_walkers(self, walkers: Any) -> Any:
        walkers, _ = orthonormalize(walkers)
        return walkers

    def on_walkers_updated(self, walkers: Any) -> None:
        walkers = tree_map(jnp.asarray, walkers)
        n_walkers = int(tree_leaves(walkers)[0].shape[0])
        self._current_walkers = walkers
        if self._n_walkers != n_walkers:
            self.bind_walkers(walkers, reinit=True)

    def _fields_to_flat(self, fields: Any) -> Array:
        leaves = tree_leaves(fields)
        return jnp.concatenate([jnp.ravel(x) for x in leaves], axis=0)

    def _fields_tree_to_flat(self, fields_tree: Any) -> Array:
        flatten_one = lambda fs: self._fields_to_flat(fs)
        return jax.vmap(jax.vmap(flatten_one))(fields_tree)

    def _precompute_bra_batch(self, fields: Any, logsw: Array) -> tuple[Any, Array]:
        eval_ansatz = jax.vmap(jax.vmap(lambda f: self.ansatz.apply(self.params, f)))
        bra_samples, bra_logw = eval_ansatz(fields)
        sample_logcoef = bra_logw - logsw
        return bra_samples, sample_logcoef

    def _refresh_flat_sample_views(self) -> None:
        if self.walker_fields_tree is None:
            self.sample_fields = None
            self.sample_logsw = None
            self.sample_log_abs = None
            self.sample_phase = None
            self.bra_samples = None
            return
        nw = int(self.walker_logsw.shape[0])
        ns = int(self.walker_logsw.shape[1])
        self.sample_fields = tree_map(
            lambda x: x.reshape((nw * ns,) + x.shape[2:]),
            self.walker_fields_tree,
        )
        self.sample_logsw = self.walker_logsw.reshape((nw * ns,))
        self.sample_log_abs = self.walker_log_abs.reshape((nw * ns,))
        self.sample_phase = self.walker_phase.reshape((nw * ns,))
        self.bra_samples = tree_map(
            lambda x: x.reshape((nw * ns,) + x.shape[2:]),
            self.walker_bra_samples,
        )

    def _update_tethered_cache(self, fields: Any, logsw: Array) -> None:
        self.walker_fields_tree = fields
        self.walker_logsw = logsw
        self.walker_fields = self._fields_tree_to_flat(fields)
        self.walker_bra_samples, sample_logcoef = self._precompute_bra_batch(fields, logsw)
        self.walker_log_abs = jnp.real(sample_logcoef)
        self.walker_phase = jnp.exp(1.0j * jnp.imag(sample_logcoef))
        self._refresh_flat_sample_views()

    def _sample_once_for_all_walkers(self, walkers: Any, state: Any):
        self._walker_rng, sample_key = jax.random.split(self._walker_rng)
        keys = jax.random.split(sample_key, self._n_walkers)

        def one_sample(key, walker, st):
            return self._sampler_batched.sample(key, (self.params, walker), st)

        new_state, data = jax.vmap(one_sample)(keys, walkers, state)
        return new_state, data

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
            raise ValueError("walkers must have a positive leading batch dimension.")

        if seed is not None:
            self._walker_rng = jax.random.PRNGKey(int(seed))
        if key is not None:
            self._walker_rng = key

        self._n_walkers = n_walkers
        self._current_walkers = walkers
        burn_steps = self.burn_in if burn_in is None else int(burn_in)
        need_init = (
            bool(reinit)
            or self._walker_sampler_state is None
            or self.walker_logsw is None
            or int(self.walker_logsw.shape[0]) != n_walkers
            or int(self.walker_logsw.shape[1]) != self.n_samples
        )

        if need_init:
            self._walker_rng, init_key = jax.random.split(self._walker_rng)
            init_keys = jax.random.split(init_key, n_walkers)
            self._walker_sampler_state = jax.vmap(
                lambda k, w: self._sampler_batched.init(k, (self.params, w))
            )(init_keys, walkers)
            if burn_steps > 0:
                self._walker_rng, burn_key = jax.random.split(self._walker_rng)
                burn_keys = jax.random.split(burn_key, n_walkers)
                self._walker_sampler_state = jax.vmap(
                    lambda k, w, st: self._sampler_batched.burn_in(
                        k, (self.params, w), st, burn_steps
                    )
                )(burn_keys, walkers, self._walker_sampler_state)

        self._walker_sampler_state, (fields, logsw) = self._sample_once_for_all_walkers(
            walkers, self._walker_sampler_state
        )
        self._update_tethered_cache(fields, logsw)

    def _ensure_tether_state(self, walkers: Any, *, reinit: bool = False) -> None:
        walkers = tree_map(jnp.asarray, walkers)
        n_walkers = int(tree_leaves(walkers)[0].shape[0])
        bad_state = (
            bool(reinit)
            or self._walker_sampler_state is None
            or self.walker_logsw is None
            or self._n_walkers != n_walkers
            or int(self.walker_logsw.shape[1]) != self.n_samples
        )
        if bad_state:
            self.bind_walkers(walkers, reinit=True)
        else:
            self._current_walkers = walkers

    def _calc_overlap_bundle(self, walkers: Any) -> tuple[Array, Array, Array]:
        def one_walker(walker, bra_samples, log_abs, phase):
            sample_sign, sample_logov = jax.vmap(lambda bra: calc_slov(bra, walker))(bra_samples)
            return _mix_overlap_terms(sample_sign, sample_logov, log_abs, phase)

        sign, logov, weights = jax.vmap(one_walker)(
            walkers,
            self.walker_bra_samples,
            self.walker_log_abs,
            self.walker_phase,
        )
        return sign, logov, weights

    def calc_slov(self, walkers: Any) -> tuple[Array, Array]:
        self._ensure_tether_state(walkers)
        sign, logov, _ = self._calc_overlap_bundle(walkers)
        return sign, logov

    def calc_local_energy(self, hamil: Any, walkers: Any) -> Array:
        self._ensure_tether_state(walkers)
        _, _, mix_weights = self._calc_overlap_bundle(walkers)

        def one_walker(walker, bra_samples, weights):
            es = jax.vmap(lambda bra: hamil.local_energy(bra, walker))(bra_samples)
            return jnp.einsum("s,s->", weights, es)

        return jax.vmap(one_walker)(walkers, self.walker_bra_samples, mix_weights)

    def calc_force_bias(self, _hamil: Any, walkers: Any, prop_data: Any) -> Array:
        self._ensure_tether_state(walkers)
        _, _, mix_weights = self._calc_overlap_bundle(walkers)

        def one_walker(walker, bra_samples, weights):
            rdms = jax.vmap(lambda bra: self._collapse_rdm(calc_rdm(bra, walker)))(bra_samples)
            rdm_mix = jnp.einsum("s,spq->pq", weights, rdms)
            return jnp.einsum("kpq,pq->k", prop_data.vhs, rdm_mix)

        return jax.vmap(one_walker)(walkers, self.walker_bra_samples, mix_weights)

    def update_tethered_samples(
        self,
        walkers: Any,
        *,
        n_steps: Optional[int] = None,
    ) -> tuple[Array, Array, Array]:
        """Advance each walker's own sample pool and return hand-off correction.

        Returns:
            handoff: phaseless correction from old/new trial-overlap ratio
            sign_new, logov_new: overlap terms after updating the sample pools
        """
        self._ensure_tether_state(walkers)
        walkers = tree_map(jnp.asarray, walkers)
        old_sign, old_logov, _ = self._calc_overlap_bundle(walkers)

        steps = self.sample_update_steps if n_steps is None else int(n_steps)
        if steps <= 0:
            ones = jnp.ones((self._n_walkers,), dtype=jnp.float64)
            return ones, old_sign, old_logov

        state = self._walker_sampler_state
        fields = self.walker_fields_tree
        logsw = self.walker_logsw
        for _ in range(steps):
            state, (fields, logsw) = self._sample_once_for_all_walkers(walkers, state)
        self._walker_sampler_state = state
        self._update_tethered_cache(fields, logsw)

        new_sign, new_logov, _ = self._calc_overlap_bundle(walkers)
        valid = jnp.isfinite(old_logov) & jnp.isfinite(new_logov)
        log_ratio = jnp.where(valid, new_logov - old_logov, -jnp.inf)
        sign_ratio = jnp.where(valid, new_sign / old_sign, 0.0 + 0.0j)
        ratio = sign_ratio * jnp.exp(log_ratio)
        handoff = _phaseless_from_ratio(ratio)
        return handoff, new_sign, new_logov

    def stochastic_reconfiguration(self, walkers: Any, weights: Array, key: Array):
        """Resample walkers and clone their tethered pools consistently."""
        self._ensure_tether_state(walkers)
        n_walkers = weights.shape[0]
        pos_weights = jnp.maximum(weights, 0.0)
        wsum = jnp.maximum(jnp.sum(pos_weights), 1.0e-12)
        probs = pos_weights / wsum
        cdf = jnp.cumsum(probs)
        u0 = jax.random.uniform(key, ())
        positions = (u0 + jnp.arange(n_walkers)) / n_walkers
        idx = jnp.searchsorted(cdf, positions)
        idx = jnp.clip(idx, 0, n_walkers - 1)

        walkers_new = tree_map(lambda x: x[idx], walkers)
        self._current_walkers = walkers_new
        self.walker_fields = self.walker_fields[idx]
        self.walker_fields_tree = tree_map(lambda x: x[idx], self.walker_fields_tree)
        self.walker_logsw = self.walker_logsw[idx]
        self.walker_log_abs = self.walker_log_abs[idx]
        self.walker_phase = self.walker_phase[idx]
        self.walker_bra_samples = tree_map(lambda x: x[idx], self.walker_bra_samples)
        self._walker_sampler_state = tree_map(lambda x: x[idx], self._walker_sampler_state)
        self._refresh_flat_sample_views()

        new_weights = jnp.ones_like(weights) * (wsum / n_walkers)
        return walkers_new, new_weights

    def refresh_samples(
        self,
        *,
        seed: Optional[int] = None,
        n_samples: Optional[int] = None,
        burn_in: Optional[int] = None,
        replace_per_refresh: Optional[int] = None,
        reinit: bool = False,
    ) -> None:
        """Compatibility wrapper: refresh tethered pools for currently bound walkers."""
        if seed is not None:
            self._walker_rng = jax.random.PRNGKey(int(seed))
        if n_samples is not None and int(n_samples) != self.n_samples:
            self.n_samples = int(n_samples)
            self._sampler_batched = make_batched(self._sampler, self.n_samples, concat=False)
            reinit = True
        if burn_in is not None:
            self.burn_in = int(burn_in)
        if replace_per_refresh is not None:
            self.replace_per_refresh = int(replace_per_refresh)

        if self._current_walkers is None:
            return
        if reinit:
            self.bind_walkers(self._current_walkers, reinit=True, burn_in=self.burn_in)
            return
        steps = int(self.replace_per_refresh)
        if steps <= 0:
            steps = max(self.sample_update_steps, 1)
        self.update_tethered_samples(self._current_walkers, n_steps=steps)

    def advance_sampler(self, steps: int, *, seed: Optional[int] = None) -> None:
        nsteps = int(steps)
        if nsteps <= 0 or self._current_walkers is None:
            return
        if seed is not None:
            self._walker_rng = jax.random.PRNGKey(int(seed))
        self.update_tethered_samples(self._current_walkers, n_steps=nsteps)

    def advance_pair_sampler(self, steps: int, *, seed: Optional[int] = None) -> None:
        """Advance pair sampler used by inference-like pure estimator."""
        nsteps = int(steps)
        if nsteps <= 0:
            return
        if self._pair_sampler is None:
            self._pair_sampler = choose_sampler_maker(self._sampler_name)(
                self._build_pair_logdens(),
                self._pair_fields_shape,
                **self._sampler_kwargs,
            )
        if self._pair_state is None:
            self._pair_rng, init_key = jax.random.split(self._pair_rng)
            self._pair_state = self._pair_sampler.init(init_key, self.params)
        if seed is not None:
            self._pair_rng = jax.random.PRNGKey(int(seed))
        self._pair_rng, burn_key = jax.random.split(self._pair_rng)
        self._pair_state = self._pair_sampler.burn_in(
            burn_key, self.params, self._pair_state, nsteps
        )

    def calc_pure_energy(self, hamil: Any, *, diag_only: bool = True) -> Array:
        """Estimate trial-only energy from current tethered sample pools."""
        if self.sample_log_abs is None or self.bra_samples is None:
            return jnp.nan

        shift = jnp.max(self.sample_log_abs)
        w = jnp.exp(self.sample_log_abs - shift)
        wsum = jnp.maximum(jnp.sum(w), 1.0e-16)

        if diag_only:
            es = jax.vmap(lambda bra: hamil.local_energy(bra, bra))(self.bra_samples)
            return jnp.sum(w * jnp.real(es)) / wsum

        alpha = jnp.exp(self.sample_log_abs + 1.0j * jnp.angle(self.sample_phase))
        n_total = int(self.sample_log_abs.shape[0])

        def one_pair(bra_s, bra_t):
            sign, logov = calc_slov(bra_s, bra_t)
            ov = sign * jnp.exp(logov)
            e_st = hamil.local_energy(bra_s, bra_t)
            return ov, e_st

        def one_s(idx_s):
            bra_s = tree_map(lambda x: x[idx_s], self.bra_samples)
            pairs = jax.vmap(
                lambda idx_t: one_pair(bra_s, tree_map(lambda x: x[idx_t], self.bra_samples))
            )(jnp.arange(n_total))
            return pairs

        ov_mat, e_mat = jax.vmap(one_s)(jnp.arange(n_total))
        coef = jnp.conj(alpha)[:, None] * alpha[None, :]
        den = jnp.sum(coef * ov_mat)
        num = jnp.sum(coef * ov_mat * e_mat)
        return jnp.real(num / den)

    def calc_pure_energy_inference(self, hamil: Any, *, n_pairs: Optional[int] = None) -> Array:
        """Inference-like pure estimator with independent bra/ket samples."""
        npairs = self.pure_eval_pairs if n_pairs is None else int(n_pairs)
        if npairs <= 0:
            return jnp.nan

        fields_pair, logsw_pair = self._sample_pairs_for_pure(npairs)
        fields_a = tree_map(lambda x: x[:, 0, ...], fields_pair)
        fields_b = tree_map(lambda x: x[:, 1, ...], fields_pair)

        eval_ansatz = jax.vmap(lambda f: self.ansatz.apply(self.params, f))
        bra, bra_lw = eval_ansatz(fields_a)
        ket, ket_lw = eval_ansatz(fields_b)
        sign, logov = jax.vmap(lambda b, k: calc_slov(b, k))(bra, ket)
        eloc = jax.vmap(lambda b, k: hamil.local_energy(b, k))(bra, ket)
        log_ratio = (logov + bra_lw + ket_lw) - logsw_pair
        shift = jnp.max(log_ratio)
        rel = jnp.exp(log_ratio - shift)
        rel = rel / jnp.maximum(jnp.mean(rel), 1.0e-16)

        num = jnp.mean((eloc * sign) * rel)
        den = jnp.mean(sign * rel)
        den_r = jnp.real(den)
        return jnp.where(jnp.abs(den_r) > 1.0e-16, jnp.real(num) / den_r, jnp.nan)

    def calc_mixed_energy_reference(self, hamil: Any) -> Array:
        """Mixed estimator using fixed ket=reference_wfn and current sample pools."""
        if self.bra_samples is None or self.sample_log_abs is None:
            return jnp.nan

        sample_sign, sample_logov = jax.vmap(
            lambda bra: calc_slov(bra, self.reference_wfn)
        )(self.bra_samples)
        sample_eloc = jax.vmap(
            lambda bra: hamil.local_energy(bra, self.reference_wfn)
        )(self.bra_samples)

        log_terms = self.sample_log_abs + sample_logov
        phases = self.sample_phase * sample_sign
        shift = jnp.max(jnp.real(log_terms))
        coeff = phases * jnp.exp(log_terms - shift)
        den = jnp.sum(coeff)
        num = jnp.sum(coeff * sample_eloc)
        return jnp.where(jnp.abs(den) > 1.0e-16, jnp.real(num / den), jnp.nan)

    def _build_pair_logdens(self):
        def _pair_logdens(params: Any, fields_pair: Any) -> Array:
            fields_a = tree_map(lambda x: x[0], fields_pair)
            fields_b = tree_map(lambda x: x[1], fields_pair)
            bra, bra_lw = self.ansatz.apply(params, fields_a)
            ket, ket_lw = self.ansatz.apply(params, fields_b)
            _, logov = calc_slov(bra, ket)
            logd = jnp.real(logov + bra_lw + ket_lw)
            return jnp.where(jnp.isfinite(logd), logd, self.logdens_floor)

        return _pair_logdens

    def _sample_pairs_for_pure(self, npairs: int):
        if self._pair_sampler is None:
            self._pair_sampler = choose_sampler_maker(self._sampler_name)(
                self._build_pair_logdens(),
                self._pair_fields_shape,
                **self._sampler_kwargs,
            )

        if self._pair_state is None:
            self._pair_rng, init_key = jax.random.split(self._pair_rng)
            state = self._pair_sampler.init(init_key, self.params)
            if self.burn_in > 0:
                self._pair_rng, burn_key = jax.random.split(self._pair_rng)
                state = self._pair_sampler.burn_in(
                    burn_key, self.params, state, self.burn_in
                )
            self._pair_state = state

        self._pair_rng, sample_key = jax.random.split(self._pair_rng)
        keys = jax.random.split(sample_key, npairs)

        def _scan_step(state, key):
            new_state, data = self._pair_sampler.sample(key, self.params, state)
            return new_state, data

        state, data = lax.scan(_scan_step, self._pair_state, keys)
        self._pair_state = state
        fields_pair, logsw_pair = data
        return fields_pair, logsw_pair
