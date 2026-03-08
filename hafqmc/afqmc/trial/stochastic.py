"""VAFQMC stochastic trial object with modularized method mixins."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import jax
import numpy as onp
from jax import numpy as jnp
from jax.tree_util import tree_leaves, tree_map

from ...ansatz import Ansatz
from ...hamiltonian import _has_spin, calc_rdm
from ...sampler import choose_sampler_maker, make_batched
from ...utils import load_pickle
from ..utils import extract_params, load_ansatz_cfg_from_hparams, normalize_ansatz_params
from .stochastic_internal_methods import (
    _build_logdens,
    _calc_overlap_bundle_from_cache,
    _collapse_rdm,
    _ensure_bound_pool,
    _evaluate_pool_bra,
    _fields_to_flat,
    _fields_tree_to_flat,
    _get_force_bias_fn,
    _get_local_energy_fn,
    _init_pool_state,
    _sample_pool_step,
    _sample_walkers_from_trial,
    _update_pool_cache,
)
from .stochastic_runtime_methods import (
    _cache_from_samples,
    _overlap_bundle_from_state,
    _pack_runtime_state,
    _require_runtime_state,
    bind_walkers,
    calc_force_bias,
    calc_force_bias_state,
    calc_local_energy,
    calc_local_energy_state,
    calc_slov,
    calc_slov_state,
    export_runtime_state,
    get_rdm1,
    import_runtime_state,
    init_runtime_state,
    init_walkers,
    measure_block_energy,
    measure_block_energy_state,
    on_walkers_updated,
    on_walkers_updated_state,
    orthonormalize_walkers,
    stochastic_reconfiguration,
    stochastic_reconfiguration_state,
    update_tethered_samples,
    update_tethered_samples_state,
)

Array = jnp.ndarray


class VAFQMCTrial:
    """Custom trial object consumed by ``hafqmc.afqmc.afqmc_energy``.

    This module keeps construction/config in this class file and moves method
    implementations into focused modules to improve readability.
    """

    def __init__(
        self,
        ansatz: Ansatz,
        params: Mapping[str, Any],
        *,
        hamiltonian: Optional[Any] = None,
        reference_wfn: Any,
        n_samples: int = 128,
        burn_in: int = 256,
        sampler_name: str = "hmc",
        sampler_kwargs: Optional[Mapping[str, Any]] = None,
        sampling_target: str = "walker_overlap",
        logdens_floor: float = -60.0,
        sample_update_steps: int = 1,
        local_energy_chunk_size: int = 0,
        init_walkers_from_trial: bool = False,
        init_walkers_burn_in: int = 0,
        init_walkers_chains_per_walker: int = 0,
        init_walkers_infer_steps: int = 10,
        max_prop: Optional[Any] = None,
        seed: int = 0,
    ) -> None:
        self.ansatz = ansatz
        self.params = params
        self.reference_wfn = tree_map(jnp.asarray, reference_wfn)
        self.nbasis = int(
            self.reference_wfn[0].shape[0] if _has_spin(self.reference_wfn) else self.reference_wfn.shape[0]
        )

        self.n_samples = int(n_samples)
        self.burn_in = int(burn_in)
        self.sample_update_steps = int(sample_update_steps)
        self.local_energy_chunk_size = int(local_energy_chunk_size)
        self.init_walkers_from_trial = bool(init_walkers_from_trial)
        self.init_walkers_burn_in = int(init_walkers_burn_in)
        self.init_walkers_chains_per_walker = int(init_walkers_chains_per_walker)
        self.init_walkers_infer_steps = int(init_walkers_infer_steps)
        self.sampling_target = str(sampling_target).lower()
        self.logdens_floor = float(logdens_floor)
        self.max_prop = max_prop
        self.seed = int(seed)
        self._hamiltonian = hamiltonian

        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if (
            self.burn_in < 0
            or self.sample_update_steps < 0
            or self.local_energy_chunk_size < 0
            or self.init_walkers_burn_in < 0
            or self.init_walkers_chains_per_walker < 0
            or self.init_walkers_infer_steps < 0
        ):
            raise ValueError("burn-in/update steps must be non-negative.")

        self.fields_shape = self.ansatz.fields_shape(self.max_prop)
        zero_fields = tree_map(
            lambda s: jnp.zeros(tuple(onp.asarray(s).tolist())),
            self.fields_shape,
        )
        self.n_fields = int(sum(int(onp.prod(x.shape)) for x in tree_leaves(zero_fields)))

        sampler_name = sampler_name.lower()
        kwargs = dict(sampler_kwargs or {})
        if sampler_name in ("hmc", "hamiltonian", "hybrid") and not kwargs:
            kwargs = {"dt": 0.1, "length": 1.0}
        self._sampler_name = sampler_name
        self._sampler_kwargs = kwargs

        self._sampler = choose_sampler_maker(self._sampler_name)(
            self._build_logdens(),
            self.fields_shape,
            **self._sampler_kwargs,
        )
        self._pool_sampler = make_batched(self._sampler, self.n_samples, concat=False)

        self._rng = jax.random.PRNGKey(self.seed)

        self._n_walkers = 0
        self._bound_walkers = None
        self._pool_state = None
        self._pool_fields_tree = None
        self._pool_logsw = None
        self._pool_log_abs = None
        self._pool_phase = None
        self._pool_bra = None

        self.walker_fields = None
        # Optional handoff cache from stage-1 init-walker sampling:
        # shape tree [n_walkers, n_samples, ...] in self.fields_shape.
        self._init_pool_fields_override = None
        self._init_walkers_infer_fn = None

        self._local_energy_fns: Dict[int, Any] = {}
        self._force_bias_fns: Dict[int, Any] = {}
        self._block_measure_fns: Dict[Tuple[int, int, int], Any] = {}

        self.rdm1 = jnp.real(self._collapse_rdm(calc_rdm(self.reference_wfn, self.reference_wfn)))

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
        sample_update_steps: int = 1,
        local_energy_chunk_size: int = 0,
        init_walkers_from_trial: bool = False,
        init_walkers_burn_in: int = 0,
        init_walkers_chains_per_walker: int = 0,
        init_walkers_infer_steps: int = 10,
        max_prop: Optional[Any] = None,
        seed: int = 0,
    ) -> "VAFQMCTrial":
        payload = load_pickle(checkpoint_path)
        params = normalize_ansatz_params(extract_params(payload))
        ansatz = Ansatz.create(hamiltonian, **dict(ansatz_cfg))
        return cls(
            ansatz,
            params,
            hamiltonian=hamiltonian,
            reference_wfn=hamiltonian.wfn0,
            n_samples=n_samples,
            burn_in=burn_in,
            sampler_name=sampler_name,
            sampler_kwargs=sampler_kwargs,
            sampling_target=sampling_target,
            logdens_floor=logdens_floor,
            sample_update_steps=sample_update_steps,
            local_energy_chunk_size=local_energy_chunk_size,
            init_walkers_from_trial=init_walkers_from_trial,
            init_walkers_burn_in=init_walkers_burn_in,
            init_walkers_chains_per_walker=init_walkers_chains_per_walker,
            init_walkers_infer_steps=init_walkers_infer_steps,
            max_prop=max_prop,
            seed=seed,
        )

    @classmethod
    def from_hparams_checkpoint(
        cls,
        hamiltonian: Any,
        checkpoint_path: str,
        *,
        hparams_path: str = "hparams.yml",
        n_samples: int = 20,
        burn_in: int = 1000,
        sampler_name: str = "hmc",
        sampler_kwargs: Optional[Mapping[str, Any]] = None,
        sampling_target: str = "walker_overlap",
        logdens_floor: float = -60.0,
        sample_update_steps: int = 1,
        local_energy_chunk_size: int = 0,
        init_walkers_from_trial: bool = False,
        init_walkers_burn_in: int = 0,
        init_walkers_chains_per_walker: int = 0,
        init_walkers_infer_steps: int = 10,
        max_prop: Optional[Any] = None,
        seed: int = 0,
    ) -> "VAFQMCTrial":
        ansatz_cfg = load_ansatz_cfg_from_hparams(hparams_path)
        return cls.from_checkpoint(
            hamiltonian,
            checkpoint_path,
            ansatz_cfg=ansatz_cfg,
            n_samples=n_samples,
            burn_in=burn_in,
            sampler_name=sampler_name,
            sampler_kwargs=sampler_kwargs,
            sampling_target=sampling_target,
            logdens_floor=logdens_floor,
            sample_update_steps=sample_update_steps,
            local_energy_chunk_size=local_energy_chunk_size,
            init_walkers_from_trial=init_walkers_from_trial,
            init_walkers_burn_in=init_walkers_burn_in,
            init_walkers_chains_per_walker=init_walkers_chains_per_walker,
            init_walkers_infer_steps=init_walkers_infer_steps,
            max_prop=max_prop,
            seed=seed,
        )

    # runtime/state APIs
    _pack_runtime_state = _pack_runtime_state
    _require_runtime_state = _require_runtime_state
    _overlap_bundle_from_state = _overlap_bundle_from_state
    _cache_from_samples = _cache_from_samples
    export_runtime_state = export_runtime_state
    import_runtime_state = import_runtime_state
    init_runtime_state = init_runtime_state
    on_walkers_updated_state = on_walkers_updated_state

    # trial hooks
    get_rdm1 = get_rdm1
    init_walkers = init_walkers
    orthonormalize_walkers = orthonormalize_walkers
    on_walkers_updated = on_walkers_updated
    bind_walkers = bind_walkers
    calc_slov = calc_slov
    calc_slov_state = calc_slov_state
    calc_local_energy = calc_local_energy
    calc_local_energy_state = calc_local_energy_state
    calc_force_bias = calc_force_bias
    calc_force_bias_state = calc_force_bias_state
    update_tethered_samples = update_tethered_samples
    update_tethered_samples_state = update_tethered_samples_state
    stochastic_reconfiguration = stochastic_reconfiguration
    stochastic_reconfiguration_state = stochastic_reconfiguration_state
    measure_block_energy = measure_block_energy
    measure_block_energy_state = measure_block_energy_state

    # internals
    _build_logdens = _build_logdens
    _collapse_rdm = _collapse_rdm
    _fields_to_flat = _fields_to_flat
    _fields_tree_to_flat = _fields_tree_to_flat
    _sample_walkers_from_trial = _sample_walkers_from_trial
    _evaluate_pool_bra = _evaluate_pool_bra
    _update_pool_cache = _update_pool_cache
    _init_pool_state = _init_pool_state
    _sample_pool_step = _sample_pool_step
    _ensure_bound_pool = _ensure_bound_pool
    _calc_overlap_bundle_from_cache = _calc_overlap_bundle_from_cache
    _get_local_energy_fn = _get_local_energy_fn
    _get_force_bias_fn = _get_force_bias_fn


__all__ = [
    "VAFQMCTrial",
    "load_ansatz_cfg_from_hparams",
]
