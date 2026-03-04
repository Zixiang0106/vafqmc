"""Fixed multi-determinant CASSCI trial for AFQMC custom driver."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import jax
import numpy as np
from jax import lax, numpy as jnp

from ...hamiltonian import calc_rdm, calc_slov
from ...molecule import get_orth_ao, rotate_wfn
from ...propagator import orthonormalize
from ...utils import load_pickle

Array = jnp.ndarray


def _to_tuple_nelec(nelec: Any) -> tuple[int, int]:
    if isinstance(nelec, (tuple, list)) and len(nelec) == 2:
        return int(nelec[0]), int(nelec[1])
    n = int(nelec)
    return n // 2, n // 2


def _bitstr_to_occ(bitstr: Any, ncas: int) -> list[int]:
    if isinstance(bitstr, str):
        value = int(bitstr, 2)
    else:
        value = int(bitstr)
    return [i for i in range(ncas) if ((value >> i) & 1) == 1]


class CASSCITrial:
    """Static multi-determinant trial built from CASSCI expansion."""

    def __init__(
        self,
        coeffs: Any,
        det_up: Any,
        det_dn: Any,
        *,
        rdm1: Optional[Any] = None,
        init_mode: str = "sample_coeff",
    ):
        coeffs = jnp.asarray(coeffs, dtype=jnp.complex128).reshape(-1)
        det_up = jnp.asarray(det_up, dtype=jnp.complex128)
        det_dn = jnp.asarray(det_dn, dtype=jnp.complex128)

        if det_up.ndim != 3 or det_dn.ndim != 3:
            raise ValueError("det_up/det_dn must have shape (n_det, n_orb, n_elec_spin).")
        if det_up.shape[0] != coeffs.shape[0] or det_dn.shape[0] != coeffs.shape[0]:
            raise ValueError("coeffs and determinants must share the same n_det.")
        if det_up.shape[1] != det_dn.shape[1]:
            raise ValueError("det_up and det_dn must share the same basis dimension.")

        self.coeffs = coeffs
        self.det_up = det_up
        self.det_dn = det_dn
        self.n_det = int(coeffs.shape[0])
        self.nbasis = int(det_up.shape[1])
        self.nelec = (int(det_up.shape[2]), int(det_dn.shape[2]))
        self.init_mode = str(init_mode).lower()

        if rdm1 is None:
            # Fallback: diagonal-in-determinant approximation for trial RDM.
            probs = jnp.abs(coeffs) ** 2
            probs = probs / jnp.maximum(jnp.sum(probs), 1.0e-16)
            rdm_diag = jax.vmap(lambda u, d: calc_rdm((u, d), (u, d)))(det_up, det_dn)
            rdm1 = jnp.einsum("d,dspq->pq", probs, rdm_diag).real
        self.rdm1 = jnp.asarray(rdm1, dtype=jnp.float64)

        probs = jnp.abs(self.coeffs)
        probs = probs / jnp.maximum(jnp.sum(probs), 1.0e-16)
        self._init_probs = probs.astype(jnp.float64)
        self._ref_idx = int(jnp.argmax(jnp.abs(self.coeffs)))

    @staticmethod
    def _parse_payload(payload: Any) -> tuple[Any, Any, Any, Any]:
        if isinstance(payload, Mapping):
            coeffs = None
            for k in ("coeffs", "coeff", "ci_coeffs"):
                if k in payload and payload[k] is not None:
                    coeffs = payload[k]
                    break
            det_up = None
            for k in ("det_up", "dets_up", "wfn_up"):
                if k in payload and payload[k] is not None:
                    det_up = payload[k]
                    break
            det_dn = None
            for k in ("det_dn", "dets_dn", "wfn_dn"):
                if k in payload and payload[k] is not None:
                    det_dn = payload[k]
                    break
            if ("dets" in payload) and payload["dets"] is not None:
                dets = payload["dets"]
                if isinstance(dets, (tuple, list)) and len(dets) == 2:
                    det_up, det_dn = dets
            rdm1 = payload.get("rdm1", payload.get("rdm", None))
            if coeffs is None or det_up is None or det_dn is None:
                raise ValueError("Invalid CASSCI trial payload: missing coeffs/det_up/det_dn.")
            return coeffs, det_up, det_dn, rdm1

        if isinstance(payload, (tuple, list)):
            if len(payload) == 3 and isinstance(payload[1], (tuple, list)) and len(payload[1]) == 2:
                return payload[0], payload[1][0], payload[1][1], payload[2]
            if len(payload) >= 3:
                rdm1 = payload[3] if len(payload) > 3 else None
                return payload[0], payload[1], payload[2], rdm1

        raise ValueError("Unsupported CASSCI trial payload format.")

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        n_det: Optional[int] = None,
        coeff_cutoff: float = 0.0,
        normalize_coeffs: bool = True,
        init_mode: str = "sample_coeff",
    ) -> "CASSCITrial":
        coeffs, det_up, det_dn, rdm1 = cls._parse_payload(payload)
        coeffs = np.asarray(coeffs).reshape(-1)
        det_up = np.asarray(det_up)
        det_dn = np.asarray(det_dn)

        if det_up.shape[0] != coeffs.shape[0] or det_dn.shape[0] != coeffs.shape[0]:
            raise ValueError("coeffs and determinant stack have inconsistent n_det.")

        keep = np.ones(coeffs.shape[0], dtype=bool)
        if coeff_cutoff > 0.0:
            keep &= np.abs(coeffs) >= float(coeff_cutoff)
        if not np.any(keep):
            raise ValueError("No determinants left after coeff_cutoff filtering.")

        coeffs = coeffs[keep]
        det_up = det_up[keep]
        det_dn = det_dn[keep]

        order = np.argsort(-np.abs(coeffs))
        coeffs = coeffs[order]
        det_up = det_up[order]
        det_dn = det_dn[order]

        if n_det is not None:
            nd = max(int(n_det), 1)
            coeffs = coeffs[:nd]
            det_up = det_up[:nd]
            det_dn = det_dn[:nd]

        if normalize_coeffs:
            norm = np.linalg.norm(coeffs)
            if norm > 0:
                coeffs = coeffs / norm

        return cls(coeffs, det_up, det_dn, rdm1=rdm1, init_mode=init_mode)

    @classmethod
    def from_file(
        cls,
        path: str,
        *,
        n_det: Optional[int] = None,
        coeff_cutoff: float = 0.0,
        normalize_coeffs: bool = True,
        init_mode: str = "sample_coeff",
    ) -> "CASSCITrial":
        payload = load_pickle(path)
        return cls.from_payload(
            payload,
            n_det=n_det,
            coeff_cutoff=coeff_cutoff,
            normalize_coeffs=normalize_coeffs,
            init_mode=init_mode,
        )

    @classmethod
    def from_pyscf_casci(
        cls,
        casci: Any,
        *,
        n_det: Optional[int] = None,
        coeff_cutoff: float = 1.0e-8,
        normalize_coeffs: bool = True,
        init_mode: str = "sample_coeff",
        orth_ao: Optional[Any] = None,
    ) -> "CASSCITrial":
        from pyscf.fci import addons

        if getattr(casci, "ci", None) is None:
            casci.kernel()

        mf = casci._scf
        X = get_orth_ao(mf, orth_ao)
        C = rotate_wfn(casci.mo_coeff, X, mf.get_ovlp())
        if C.ndim != 2:
            raise ValueError("Only restricted-orbital CASCI is supported in CASSCITrial.")

        ncore = int(casci.ncore)
        ncas = int(casci.ncas)
        neleca, nelecb = _to_tuple_nelec(casci.nelecas)
        nmo = int(C.shape[1])

        large = addons.large_ci(
            casci.ci,
            ncas,
            casci.nelecas,
            tol=float(coeff_cutoff),
            return_strs=True,
        )
        if len(large) == 0:
            raise ValueError("No CASCI determinants found for requested coeff_cutoff.")

        large = sorted(large, key=lambda x: abs(x[0]), reverse=True)
        if n_det is not None:
            large = large[: max(int(n_det), 1)]

        core_occ = list(range(ncore))
        coeffs = []
        det_up = []
        det_dn = []
        for coeff, astr, bstr in large:
            a_occ_cas = _bitstr_to_occ(astr, ncas)
            b_occ_cas = _bitstr_to_occ(bstr, ncas)
            if len(a_occ_cas) != neleca or len(b_occ_cas) != nelecb:
                continue
            a_occ = core_occ + [ncore + i for i in a_occ_cas]
            b_occ = core_occ + [ncore + i for i in b_occ_cas]
            coeffs.append(complex(coeff))
            det_up.append(C[:, a_occ])
            det_dn.append(C[:, b_occ])

        if len(coeffs) == 0:
            raise ValueError("No valid CASCI determinants survived occupancy checks.")

        # Build spin-summed full-space RDM from CASCI active-space 1RDM.
        rdm1_act = casci.fcisolver.make_rdm1(casci.ci, ncas, casci.nelecas)
        if isinstance(rdm1_act, (tuple, list)) and len(rdm1_act) == 2:
            rdm1_act = np.asarray(rdm1_act[0]) + np.asarray(rdm1_act[1])
        rdm1_act = np.asarray(rdm1_act)
        rdm1_mo = np.zeros((nmo, nmo), dtype=np.result_type(rdm1_act.dtype, C.dtype))
        if ncore > 0:
            rdm1_mo[:ncore, :ncore] = 2.0 * np.eye(ncore, dtype=rdm1_mo.dtype)
        rdm1_mo[ncore : ncore + ncas, ncore : ncore + ncas] = rdm1_act
        rdm1 = C @ rdm1_mo @ C.conj().T

        return cls.from_payload(
            {
                "coeffs": np.asarray(coeffs),
                "det_up": np.asarray(det_up),
                "det_dn": np.asarray(det_dn),
                "rdm1": rdm1.real,
            },
            n_det=n_det,
            coeff_cutoff=0.0,
            normalize_coeffs=normalize_coeffs,
            init_mode=init_mode,
        )

    def get_rdm1(self) -> Array:
        return self.rdm1

    def _det_slov(self, walkers: Any) -> tuple[Array, Array]:
        w_up, w_dn = walkers

        def one_det(du, dd):
            return jax.vmap(lambda wu, wd: calc_slov((du, dd), (wu, wd)))(w_up, w_dn)

        return jax.vmap(one_det)(self.det_up, self.det_dn)

    def _mix_from_det_slov(self, sign_det: Array, log_det: Array) -> tuple[Array, Array, Array]:
        coeff_abs = jnp.abs(self.coeffs)
        safe_abs = jnp.where(coeff_abs > 0.0, coeff_abs, 1.0)
        coeff_phase = jnp.where(coeff_abs > 0.0, self.coeffs / safe_abs, 0.0 + 0.0j)

        log_terms = log_det + jnp.log(safe_abs)[:, None]
        phase_terms = sign_det * coeff_phase[:, None]

        finite = jnp.isfinite(log_terms)
        any_finite = jnp.any(finite, axis=0, keepdims=True)
        shift = jnp.max(jnp.where(finite, jnp.real(log_terms), -jnp.inf), axis=0, keepdims=True)
        shift = jnp.where(any_finite, shift, 0.0)
        scaled_mag = jnp.where(finite, jnp.exp(log_terms - shift), 0.0)
        scaled = phase_terms * scaled_mag
        total = jnp.sum(scaled, axis=0)
        total_abs = jnp.abs(total)

        sign = jnp.where(total_abs > 0.0, total / total_abs, 0.0 + 0.0j)
        logov = jnp.where(total_abs > 0.0, jnp.squeeze(shift, axis=0) + jnp.log(total_abs), -jnp.inf)

        safe_total = jnp.where(total_abs > 0.0, total, 1.0 + 0.0j)
        mix_w = jnp.where(total_abs[None, :] > 0.0, scaled / safe_total[None, :], 0.0 + 0.0j)
        return sign, logov, mix_w

    def calc_slov(self, walkers: Any) -> tuple[Array, Array]:
        sign_det, log_det = self._det_slov(walkers)
        sign, logov, _ = self._mix_from_det_slov(sign_det, log_det)
        return sign, logov

    def calc_overlap(self, walkers: Any) -> Array:
        sign, logov = self.calc_slov(walkers)
        amp = jnp.where(jnp.isfinite(logov), jnp.exp(logov), 0.0)
        return sign * amp

    def calc_rdm(self, walkers: Any) -> Array:
        w_up, w_dn = walkers
        sign_det, log_det = self._det_slov(walkers)
        _, _, mix_w = self._mix_from_det_slov(sign_det, log_det)
        valid = jnp.isfinite(log_det)
        zero_rdm = jnp.zeros((2, self.nbasis, self.nbasis), dtype=jnp.complex128)

        def one_det(du, dd, v_det):
            def one_w(wu, wd, v):
                return lax.cond(v, lambda _: calc_rdm((du, dd), (wu, wd)), lambda _: zero_rdm, operand=None)

            return jax.vmap(one_w)(w_up, w_dn, v_det)

        rdm_det = jax.vmap(one_det)(self.det_up, self.det_dn, valid)
        return jnp.einsum("dw,dwspq->wspq", mix_w, rdm_det)

    def calc_local_energy(self, hamil: Any, walkers: Any) -> Array:
        w_up, w_dn = walkers
        sign_det, log_det = self._det_slov(walkers)
        _, _, mix_w = self._mix_from_det_slov(sign_det, log_det)
        valid = jnp.isfinite(log_det)
        zero_e = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)

        def one_det(du, dd, v_det):
            def one_w(wu, wd, v):
                return lax.cond(v, lambda _: hamil.local_energy((du, dd), (wu, wd)), lambda _: zero_e, operand=None)

            return jax.vmap(one_w)(w_up, w_dn, v_det)

        e_det = jax.vmap(one_det)(self.det_up, self.det_dn, valid)
        return jnp.einsum("dw,dw->w", mix_w, e_det)

    def init_walkers(self, n_walkers: int, key: Array, noise: float = 0.0) -> Any:
        n_walkers = int(n_walkers)
        if self.init_mode in ("reference", "ref", "max_coeff"):
            idx = jnp.full((n_walkers,), self._ref_idx, dtype=jnp.int32)
        else:
            idx = jax.random.choice(
                key,
                self.n_det,
                shape=(n_walkers,),
                replace=True,
                p=self._init_probs,
            )
        w_up = self.det_up[idx]
        w_dn = self.det_dn[idx]

        if noise > 0.0:
            key, k1, k2 = jax.random.split(key, 3)
            w_up = w_up + noise * jax.random.normal(k1, w_up.shape)
            w_dn = w_dn + noise * jax.random.normal(k2, w_dn.shape)

        walkers, _ = orthonormalize((w_up, w_dn))
        return walkers


__all__ = ["CASSCITrial"]
