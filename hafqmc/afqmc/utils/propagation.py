"""AFQMC propagation kernels and batch estimator helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax
from jax import numpy as jnp
from jax import scipy as jsp

from ...hamiltonian import Hamiltonian, calc_rdm, calc_slov, calc_v0
from ...utils import DEFAULT_EXPM
from .core import Array, Wfn, _is_custom_trial, _require_spin_det, _spin_sum_rdm


@dataclass
class PropagationData:
    hmf: Array
    exp_h1: Array
    vhs: Array
    mf_shifts: Array
    h0_prop: Array
    dt: float
    sqrt_dt: float


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
    if isinstance(walkers, (tuple, list)):
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

    walkers_ghf = walkers
    nrow = int(walkers_ghf.shape[1])
    nao = int(prop_data.exp_h1.shape[0])

    if nrow not in (nao, 2 * nao):
        raise ValueError(
            f"Unsupported GHF walker row dimension {nrow}; expected {nao} or {2 * nao}."
        )

    exp_h1 = prop_data.exp_h1
    vhs_sum = jnp.tensordot(shifted_fields, prop_data.vhs, axes=1)  # [w, nao, nao]

    if nrow == 2 * nao:
        z_h1 = jnp.zeros_like(exp_h1)
        exp_h1 = jnp.block([[exp_h1, z_h1], [z_h1, exp_h1]])

        z_v = jnp.zeros_like(vhs_sum)
        top = jnp.concatenate([vhs_sum, z_v], axis=2)
        bot = jnp.concatenate([z_v, vhs_sum], axis=2)
        vhs_sum = jnp.concatenate([top, bot], axis=1)

    walkers_ghf = jnp.einsum("ij,wjk->wik", exp_h1, walkers_ghf)
    vhs_sum = 1.0j * prop_data.sqrt_dt * vhs_sum
    if not jnp.issubdtype(vhs_sum.dtype, jnp.complexfloating):
        vhs_sum = vhs_sum.astype(jnp.complex128)
    walkers_ghf = walkers_ghf.astype(vhs_sum.dtype)
    walkers_ghf = jax.vmap(lambda a, b: DEFAULT_EXPM(a, b))(vhs_sum, walkers_ghf)
    walkers_ghf = jnp.einsum("ij,wjk->wik", exp_h1, walkers_ghf)
    return walkers_ghf


__all__ = [
    "PropagationData",
    "apply_trotter",
    "build_propagation_data",
    "calc_force_bias",
    "calc_local_energy_batch",
    "calc_rdm_batch",
    "calc_slov_batch",
]
