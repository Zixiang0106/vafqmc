import argparse
import numpy as np
from pyscf import gto, scf

from hafqmc.afqmc.afqmc_utils import build_hamiltonian_pickle
from hafqmc.utils import load_pickle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build AFQMC hamiltonian.pkl from a UHF reference."
    )
    parser.add_argument(
        "--atom",
        type=str,
        default="N 0 0 0; N 0 0 3.6",
        help="PySCF atom string.",
    )
    parser.add_argument("--basis", type=str, default="ccpvdz")
    parser.add_argument("--unit", type=str, default="B")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0, help="2S = Nalpha - Nbeta")
    parser.add_argument("--out", type=str, default="hamiltonian.pkl")
    parser.add_argument("--chol-cut", type=float, default=1e-6)
    parser.add_argument(
        "--break-sym",
        type=float,
        default=1e-3,
        help="Small diagonal perturbation to alpha/beta initial DM to avoid collapsing to RHF.",
    )
    parser.add_argument("--max-cycle", type=int, default=200)
    parser.add_argument("--verbose", type=int, default=4)
    return parser.parse_args()


def run_uhf_with_optional_symmetry_break(mol, break_sym=1e-3, max_cycle=200):
    mf = scf.UHF(mol)
    mf.max_cycle = max_cycle

    if break_sym > 0:
        dm = mf.get_init_guess(mol=mol, key="minao")
        if dm.ndim == 2:
            dm = np.stack([dm.copy(), dm.copy()], axis=0)
        nao = dm.shape[-1]
        shift = break_sym * np.eye(nao)
        dm[0] = dm[0] + shift
        dm[1] = dm[1] - shift
        mf.kernel(dm0=dm)
    else:
        mf.kernel()

    return mf


def main():
    args = parse_args()

    mol = gto.M(
        atom=args.atom,
        basis=args.basis,
        unit=args.unit,
        charge=args.charge,
        spin=args.spin,
        verbose=args.verbose,
    )

    mf = run_uhf_with_optional_symmetry_break(
        mol, break_sym=args.break_sym, max_cycle=args.max_cycle
    )

    if not mf.converged:
        raise RuntimeError("UHF did not converge. Try larger --max-cycle or different --break-sym.")

    build_hamiltonian_pickle(
        mf,
        args.out,
        chol_cut=args.chol_cut,
        orth_ao=None,
        full_eri=False,
        with_cc=False,
        with_ghf=False,
    )

    h1e, ceri, enuc, wfn0, aux = load_pickle(args.out)
    up, dn = wfn0
    diff = float(np.max(np.abs(up - dn)))

    print(f"Saved: {args.out}")
    print(f"UHF e_tot = {mf.e_tot:.12f}")
    print(f"max|wfn0_up - wfn0_dn| = {diff:.6e}")
    if diff < 1e-8:
        print("WARNING: UHF collapsed to restricted-like solution (up ~= dn).")


if __name__ == "__main__":
    main()
