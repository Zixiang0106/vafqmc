"""VAFQMC trial adapter for AFQMC custom-trial interface.

This module provides a Python-side replacement for the C++ trial logic:
it samples auxiliary fields (default: HMC), builds a stochastic expansion of
the variational trial state, and exposes overlap / force-bias / local-energy
evaluators required by ``hafqmc.afqmc``.
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
from ..sampler import choose_sampler_maker
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

    # Fallback recursive search.
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
    """Match the tuple-unpack logic used in ``hafqmc.train``.

    In ``train.py``:
        params = load_pickle(path)
        if isinstance(params, tuple): params = params[1]
        if isinstance(params, tuple): params = params[1]
    """
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

    max_log = jnp.max(log_terms)
    scaled = phases * jnp.exp(log_terms - max_log)
    total = jnp.sum(scaled)
    total_abs = jnp.abs(total)
    valid = total_abs > 0

    sign = jnp.where(valid, total / total_abs, 0.0 + 0.0j)
    logov = jnp.where(valid, max_log + jnp.log(total_abs), -jnp.inf)
    weights = jnp.where(valid, scaled / total, jnp.zeros_like(scaled))
    return sign, logov, weights


class VAFQMCTrial:
    """Custom trial object consumed by ``hafqmc.afqmc.afqmc_energy``.

    The trial state is represented by a stochastic expansion:
    ``|Psi_T> ~ sum_s c_s |phi_s>``, where ``|phi_s>`` is produced by the
    variational ansatz for sampled auxiliary fields.
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
        sampling_target: str = "coeff_overlap",
        logdens_floor: float = -60.0,
        refresh_interval: int = 0,
        replace_per_refresh: int = 1,
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
        self.pure_eval_pairs = int(pure_eval_pairs)
        self.max_prop = max_prop
        self.seed = int(seed)
        self.fields_shape = self.ansatz.fields_shape(self.max_prop)

        name = sampler_name.lower()
        kwargs = dict(sampler_kwargs or {})
        if name in ("hmc", "hamiltonian", "hybrid") and not kwargs:
            kwargs = {"dt": 0.1, "length": 1.0}
        self._sampler_name = name
        self._sampler_kwargs = kwargs
        self._sampler = choose_sampler_maker(name)(self._build_logdens(), self.fields_shape, **kwargs)
        self._rng = jax.random.PRNGKey(self.seed)
        self._sampler_state = None

        # Pair sampler for inference-like pure estimator, lazily initialized.
        # Keep leaves as ndarray shapes (not Python tuples), otherwise pytree
        # utilities recurse into shape entries and break sampler dimensions.
        self._pair_fields_shape = tree_map(
            lambda s: onp.array((2, *tuple(onp.asarray(s).tolist()))),
            self.fields_shape,
        )
        self._pair_sampler = None
        self._pair_state = None
        self._pair_rng = jax.random.PRNGKey(self.seed + 7919)

        self.sample_fields = None
        self.sample_logsw = None
        self.sample_log_abs = None
        self.sample_phase = None
        self.bra_samples = None
        self.rdm1 = None
        self._ring_pos = 0

        self.refresh_samples(reinit=True)

    def _build_logdens(self):
        if self.sampling_target in ("gaussian", "normal", "std"):
            return _gaussian_logdens

        # Importance distribution guided by trial coefficient magnitude wrt reference.
        def _coeff_overlap_logdens(params: Any, fields: Any) -> Array:
            bra, bra_lw = self.ansatz.apply(params, fields)
            _, logov_ref = calc_slov(bra, self.reference_wfn)
            logd = jnp.real(logov_ref + bra_lw)
            return jnp.where(jnp.isfinite(logd), logd, self.logdens_floor)

        return _coeff_overlap_logdens

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
        sampling_target: str = "coeff_overlap",
        logdens_floor: float = -60.0,
        refresh_interval: int = 0,
        replace_per_refresh: int = 1,
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
            pure_eval_pairs=pure_eval_pairs,
            max_prop=max_prop,
            seed=seed,
        )

    def refresh_samples(
        self,
        *,
        seed: Optional[int] = None,
        n_samples: Optional[int] = None,
        burn_in: Optional[int] = None,
        replace_per_refresh: Optional[int] = None,
        reinit: bool = False,
    ) -> None:
        if seed is not None:
            self._rng = jax.random.PRNGKey(int(seed))
        if n_samples is not None:
            self.n_samples = int(n_samples)
        if burn_in is not None:
            self.burn_in = int(burn_in)
        if replace_per_refresh is not None:
            self.replace_per_refresh = int(replace_per_refresh)
        if self.n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")

        full_reset = (
            bool(reinit)
            or self._sampler_state is None
            or self.sample_fields is None
            or self.sample_logsw is None
        )
        if not full_reset and int(self.sample_logsw.shape[0]) != self.n_samples:
            full_reset = True

        if full_reset:
            self._rng, init_key = jax.random.split(self._rng)
            state = self._sampler.init(init_key, self.params)
            if self.burn_in > 0:
                self._rng, burn_key = jax.random.split(self._rng)
                state = self._sampler.burn_in(burn_key, self.params, state, self.burn_in)

            self._rng, sample_key = jax.random.split(self._rng)
            keys = jax.random.split(sample_key, self.n_samples)
            fields_buf = []
            logsw_buf = []
            for k in keys:
                state, (fields, logsw) = self._sampler.sample(k, self.params, state)
                fields_buf.append(fields)
                logsw_buf.append(logsw)
            self._sampler_state = state
            self.sample_fields = tree_map(lambda *xs: jnp.stack(xs), *fields_buf)
            self.sample_logsw = jnp.stack(logsw_buf)
            self._ring_pos = 0
        else:
            replace_count = int(self.replace_per_refresh)
            if replace_count <= 0:
                replace_count = self.n_samples
            replace_count = min(replace_count, self.n_samples)
            self._rng, sample_key = jax.random.split(self._rng)
            keys = jax.random.split(sample_key, replace_count)
            state = self._sampler_state
            for i, k in enumerate(keys):
                state, (fields, logsw) = self._sampler.sample(k, self.params, state)
                idx = (self._ring_pos + i) % self.n_samples
                self.sample_fields = tree_map(
                    lambda arr, val: arr.at[idx].set(val),
                    self.sample_fields,
                    fields,
                )
                self.sample_logsw = self.sample_logsw.at[idx].set(logsw)
            self._sampler_state = state
            self._ring_pos = (self._ring_pos + replace_count) % self.n_samples

        self.bra_samples, sample_logcoef = self._precompute_bra(self.sample_fields, self.sample_logsw)
        self.sample_log_abs = jnp.real(sample_logcoef)
        self.sample_phase = jnp.exp(1.0j * jnp.imag(sample_logcoef))
        self.rdm1 = self._estimate_rdm1()

    def advance_sampler(self, steps: int, *, seed: Optional[int] = None) -> None:
        """Advance the inner sampler state without rebuilding trial samples."""
        nsteps = int(steps)
        if nsteps <= 0:
            return
        if self._sampler_state is None:
            self._rng, init_key = jax.random.split(self._rng)
            self._sampler_state = self._sampler.init(init_key, self.params)
        if seed is not None:
            self._rng = jax.random.PRNGKey(int(seed))
        self._rng, burn_key = jax.random.split(self._rng)
        self._sampler_state = self._sampler.burn_in(
            burn_key, self.params, self._sampler_state, nsteps
        )

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

    def _precompute_bra(self, fields: Any, logsw: Array) -> tuple[Any, Array]:
        eval_ansatz = jax.vmap(lambda f: self.ansatz.apply(self.params, f))
        bra_samples, bra_logw = eval_ansatz(fields)
        sample_logcoef = bra_logw - logsw
        return bra_samples, sample_logcoef

    def _estimate_rdm1(self) -> Array:
        calc_one = lambda bra: self._collapse_rdm(calc_rdm(bra, bra))
        rdm_samples = jax.vmap(calc_one)(self.bra_samples)
        # Diagonal approximation with positive weights for MF shift setup.
        shift = jnp.max(self.sample_log_abs)
        weights = jnp.exp(self.sample_log_abs - shift)
        wsum = jnp.maximum(jnp.sum(weights), 1.0e-16)
        rdm1 = jnp.einsum("s,spq->pq", weights, rdm_samples) / wsum
        return jnp.real(rdm1)

    def _collapse_rdm(self, rdm: Array) -> Array:
        """Convert spin-resolved/GHF RDM into spin-summed spatial RDM."""
        if rdm.ndim == 2 and rdm.shape[-1] == 2 * self.nbasis:
            # GHF block layout -> sum alpha-alpha and beta-beta blocks.
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

    def _walker_overlap_weights(self, walker: Any) -> tuple[Array, Array, Array]:
        sample_sign, sample_logov = jax.vmap(lambda bra: calc_slov(bra, walker))(self.bra_samples)
        return _mix_overlap_terms(sample_sign, sample_logov, self.sample_log_abs, self.sample_phase)

    def calc_slov(self, walkers: Any) -> tuple[Array, Array]:
        per_walker = lambda w: self._walker_overlap_weights(w)[:2]
        sign, logov = jax.vmap(per_walker)(walkers)
        return sign, logov

    def calc_local_energy(self, hamil: Any, walkers: Any) -> Array:
        def one_walker(walker):
            _, _, weights = self._walker_overlap_weights(walker)
            es = jax.vmap(lambda bra: hamil.local_energy(bra, walker))(self.bra_samples)
            return jnp.einsum("s,s->", weights, es)

        return jax.vmap(one_walker)(walkers)

    def calc_pure_energy(self, hamil: Any, *, diag_only: bool = True) -> Array:
        """Estimate trial-only energy from current sample pool.

        By default this uses a diagonal estimator:
            sum_s w_s * <phi_s|H|phi_s> / sum_s w_s
        where w_s ~ |coeff_s|.
        This is inexpensive and useful for burn-in diagnostics.
        """
        shift = jnp.max(self.sample_log_abs)
        w = jnp.exp(self.sample_log_abs - shift)
        wsum = jnp.maximum(jnp.sum(w), 1.0e-16)

        if diag_only:
            es = jax.vmap(lambda bra: hamil.local_energy(bra, bra))(self.bra_samples)
            return jnp.sum(w * jnp.real(es)) / wsum

        # Full pair estimator:
        # E = (sum_{s,t} a_s* a_t <s|t> E_st) / (sum_{s,t} a_s* a_t <s|t>)
        # Expensive (O(n_samples^2)); keep for debugging only.
        alpha = jnp.exp(self.sample_log_abs + 1.0j * jnp.angle(self.sample_phase))

        def one_pair(bra_s, bra_t):
            sign, logov = calc_slov(bra_s, bra_t)
            ov = sign * jnp.exp(logov)
            e_st = hamil.local_energy(bra_s, bra_t)
            return ov, e_st

        def one_s(idx_s):
            bra_s = tree_map(lambda x: x[idx_s], self.bra_samples)
            pairs = jax.vmap(lambda idx_t: one_pair(bra_s, tree_map(lambda x: x[idx_t], self.bra_samples)))(
                jnp.arange(self.n_samples)
            )
            return pairs

        ov_mat, e_mat = jax.vmap(one_s)(jnp.arange(self.n_samples))
        a_conj = jnp.conj(alpha)[:, None]
        a = alpha[None, :]
        coef = a_conj * a
        den = jnp.sum(coef * ov_mat)
        num = jnp.sum(coef * ov_mat * e_mat)
        return jnp.real(num / den)

    def calc_pure_energy_inference(self, hamil: Any, *, n_pairs: Optional[int] = None) -> Array:
        """Inference-like pure estimator with independent bra/ket samples.

        This mirrors the energy ratio used in ``hafqmc.estimator.make_eval_total``:
            E = Re[ <(eloc * sign) * w> / <sign * w> ],
        where w is the importance ratio between overlap measure and sampling density.
        """
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
        """Mixed estimator using fixed ket=reference_wfn and sampled bra expansion."""
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
                state = self._pair_sampler.burn_in(burn_key, self.params, state, self.burn_in)
            self._pair_state = state

        self._pair_rng, sample_key = jax.random.split(self._pair_rng)
        keys = jax.random.split(sample_key, npairs)

        # Vectorize Markov-chain advancement with scan to avoid Python-loop overhead.
        def _scan_step(state, key):
            new_state, data = self._pair_sampler.sample(key, self.params, state)
            return new_state, data

        state, data = lax.scan(_scan_step, self._pair_state, keys)
        self._pair_state = state
        fields_pair, logsw_pair = data
        return fields_pair, logsw_pair

    def calc_force_bias(self, _hamil: Any, walkers: Any, prop_data: Any) -> Array:
        def one_walker(walker):
            _, _, weights = self._walker_overlap_weights(walker)
            rdms = jax.vmap(lambda bra: self._collapse_rdm(calc_rdm(bra, walker)))(self.bra_samples)
            rdm_mix = jnp.einsum("s,spq->pq", weights, rdms)
            return jnp.einsum("kpq,pq->k", prop_data.vhs, rdm_mix)

        return jax.vmap(one_walker)(walkers)
