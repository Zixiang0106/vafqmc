"""AFQMC utilities and trial hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import jax
import numpy as onp
from jax import numpy as jnp
from jax import scipy as jsp

from ..hamiltonian import Hamiltonian, calc_rdm, calc_slov, calc_v0, _has_spin
from ..utils import DEFAULT_EXPM, load_pickle, save_pickle

Array = jnp.ndarray
SpinWfn = Tuple[Array, Array]
Wfn = Union[Array, SpinWfn]


# Trial interface (for VAFQMC trial integration):
# - calc_slov(walkers) -> (sign, log|ov|)
# - calc_force_bias(hamil, walkers, prop_data) -> force_bias
# - calc_local_energy(hamil, walkers) -> energies
# - get_rdm1() or calc_rdm1() or rdm1 -> spin-summed RDM for mean-field shift
# - init_walkers(n_walkers, key, noise=0.0) -> walkers (optional)
# - orthonormalize_walkers(walkers) -> walkers (optional)


@dataclass
class PropagationData:
    hmf: Array
    exp_h1: Array
    vhs: Array
    mf_shifts: Array
    h0_prop: Array
    dt: float
    sqrt_dt: float


def _is_det_trial(trial: Any) -> bool:
    if isinstance(trial, (tuple, list)):
        return True
    return isinstance(trial, (jnp.ndarray, onp.ndarray))


def _is_custom_trial(trial: Any) -> bool:
    return not _is_det_trial(trial)


def _require_spin_det(trial: Wfn) -> None:
    if not _has_spin(trial):
        raise ValueError(
            "AFQMC expects spin-separated (up, down) determinant trial for "
            "built-in determinant mode."
        )


def _spin_sum_rdm(rdm: Array) -> Array:
    if rdm.ndim == 3:
        return rdm[0] + rdm[1]
    if rdm.ndim == 4:
        return rdm.sum(1)
    return rdm


def _get_trial_rdm1(trial: Any) -> Array:
    if _is_custom_trial(trial):
        if hasattr(trial, "get_rdm1"):
            return trial.get_rdm1()
        if hasattr(trial, "calc_rdm1"):
            return trial.calc_rdm1()
        if hasattr(trial, "rdm1"):
            return trial.rdm1
        raise ValueError("Custom trial must provide get_rdm1/calc_rdm1/rdm1.")
    _require_spin_det(trial)
    rdm_t = calc_rdm(trial, trial)
    return _spin_sum_rdm(rdm_t)


def build_propagation_data(hamil: Hamiltonian, trial: Any, dt: float) -> PropagationData:
    h1 = hamil.h1e
    vhs = hamil.ceri
    v0 = 0.5 * calc_v0(vhs)

    rdm_t = _get_trial_rdm1(trial)
    vbar = jnp.einsum("kpq,pq->k", vhs, rdm_t)
    v1 = jnp.einsum("k,kpq->pq", vbar, vhs)

    h1_mod = h1 - v0 + v1
    mf_shifts = 1.0j * vbar
    h0_prop = hamil.enuc + 0.5 * jnp.sum(mf_shifts * mf_shifts)
    exp_h1 = jsp.linalg.expm(-0.5 * dt * h1_mod)

    return PropagationData(
        hmf=h1_mod,
        exp_h1=exp_h1,
        vhs=vhs,
        mf_shifts=mf_shifts,
        h0_prop=h0_prop,
        dt=dt,
        sqrt_dt=jnp.sqrt(dt),
    )


def calc_slov_batch(trial: Any, walkers: Any) -> Tuple[Array, Array]:
    if _is_custom_trial(trial):
        if hasattr(trial, "calc_slov"):
            return trial.calc_slov(walkers)
        if hasattr(trial, "calc_overlap"):
            ov = trial.calc_overlap(walkers)
            mag = jnp.abs(ov)
            sign = jnp.where(mag > 0, ov / mag, 0.0 + 0.0j)
            logov = jnp.where(mag > 0, jnp.log(mag), -jnp.inf)
            return sign, logov
        raise ValueError("Custom trial must provide calc_slov or calc_overlap.")

    _require_spin_det(trial)
    w_up, w_dn = walkers

    def single(wu, wd):
        return calc_slov(trial, (wu, wd))

    return jax.vmap(single)(w_up, w_dn)


def calc_rdm_batch(trial: Any, walkers: Any) -> Array:
    if _is_custom_trial(trial):
        if hasattr(trial, "calc_rdm"):
            return trial.calc_rdm(walkers)
        raise ValueError("Custom trial must provide calc_rdm for RDM evaluation.")

    _require_spin_det(trial)
    w_up, w_dn = walkers

    def single(wu, wd):
        return calc_rdm(trial, (wu, wd))

    return jax.vmap(single)(w_up, w_dn)


def calc_local_energy_batch(hamil: Hamiltonian, trial: Any, walkers: Any) -> Array:
    if _is_custom_trial(trial):
        if hasattr(trial, "calc_local_energy"):
            return trial.calc_local_energy(hamil, walkers)
        if hasattr(trial, "calc_energy"):
            return trial.calc_energy(hamil, walkers)
        raise ValueError("Custom trial must provide calc_local_energy or calc_energy.")

    _require_spin_det(trial)
    w_up, w_dn = walkers

    def single(wu, wd):
        return hamil.local_energy(trial, (wu, wd))

    return jax.vmap(single)(w_up, w_dn)


def calc_force_bias(hamil: Hamiltonian, trial: Any, walkers: Any, prop_data: PropagationData) -> Array:
    if _is_custom_trial(trial) and hasattr(trial, "calc_force_bias"):
        return trial.calc_force_bias(hamil, walkers, prop_data)

    rdm = calc_rdm_batch(trial, walkers)
    rdm_sum = _spin_sum_rdm(rdm)
    return jnp.einsum("kpq,wpq->wk", prop_data.vhs, rdm_sum)


def apply_trotter(walkers: Wfn, shifted_fields: Array, prop_data: PropagationData) -> Wfn:
    _require_spin_det(walkers)
    w_up, w_dn = walkers

    w_up = jnp.einsum("ij,wjk->wik", prop_data.exp_h1, w_up)
    w_dn = jnp.einsum("ij,wjk->wik", prop_data.exp_h1, w_dn)

    vhs_sum = jnp.tensordot(shifted_fields, prop_data.vhs, axes=1)
    vhs_sum = 1.0j * prop_data.sqrt_dt * vhs_sum
    if not jnp.issubdtype(vhs_sum.dtype, jnp.complexfloating):
        vhs_sum = vhs_sum.astype(jnp.complex128)
    w_up = w_up.astype(vhs_sum.dtype)
    w_dn = w_dn.astype(vhs_sum.dtype)
    w_up = jax.vmap(lambda a, b: DEFAULT_EXPM(a, b))(vhs_sum, w_up)
    w_dn = jax.vmap(lambda a, b: DEFAULT_EXPM(a, b))(vhs_sum, w_dn)

    w_up = jnp.einsum("ij,wjk->wik", prop_data.exp_h1, w_up)
    w_dn = jnp.einsum("ij,wjk->wik", prop_data.exp_h1, w_dn)

    return (w_up, w_dn)


def save_hamiltonian(hamil: Hamiltonian, path: str) -> None:
    save_pickle(path, hamil.to_tuple())


def load_hamiltonian(path: str) -> Hamiltonian:
    h1e, ceri, enuc, wfn0, aux = load_pickle(path)
    return Hamiltonian(h1e, ceri, enuc, wfn0, aux)


def build_hamiltonian_pickle(
    mf: Any,
    path: str,
    *,
    chol_cut: float = 1e-6,
    orth_ao: Optional[Any] = None,
    full_eri: bool = False,
    with_cc: bool = False,
    with_ghf: bool = False,
) -> Hamiltonian:
    hamil = Hamiltonian.from_pyscf(
        mf,
        chol_cut=chol_cut,
        orth_ao=orth_ao,
        full_eri=full_eri,
        with_cc=with_cc,
        with_ghf=with_ghf,
    )
    save_hamiltonian(hamil, path)
    return hamil
