from pyscf import gto, mcscf, scf

from hafqmc.afqmc import AFQMCConfig, CASSCITrial, afqmc_energy
from hafqmc.hamiltonian import Hamiltonian


# =========================
# System: N2 @ 3.6 Bohr, cc-pVDZ
# =========================
mol = gto.M(
    atom="""
    N 0.0 0.0 0.0
    N 3.6 0.0 0.0
    """,
    basis="ccpvdz",
    unit="B",
    spin=0,
    verbose=4,
)


# =========================
# RHF (with stability refinement)
# =========================
mf = scf.UHF(mol)
mf.kernel()
for _ in range(5):
    mo1 = mf.stability()[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.run(dm1)
    mf.stability()
print("Total RHF energy = %.12f" % (mf.energy_tot()))


# =========================
# Build AFQMC Hamiltonian
# =========================
hamil = Hamiltonian.from_pyscf(
    mf,
    chol_cut=1.0e-6,
    orth_ao=None,
    full_eri=False,
)


# =========================
# Build fixed CASCI multi-det trial
# =========================
# You can tune these:
# - ncas/nelecas: active space size
# - n_det: number of determinants kept in trial
casci = mcscf.CASCI(mf, ncas=8, nelecas=8)
casci.kernel()

trial = CASSCITrial.from_pyscf_casci(
    casci,
    n_det=20,
    coeff_cutoff=1.0e-8,
    normalize_coeffs=True,
    init_mode="sample_coeff",
    orth_ao=None,
)


# =========================
# AFQMC run config
# =========================
cfg = AFQMCConfig.cassci_example()
cfg.seed = 0

cfg.propagation.dt = 0.005
cfg.propagation.n_walkers = 200
cfg.propagation.n_eq_steps = 20
cfg.propagation.n_blocks = 200
cfg.propagation.n_prop_steps = 50
cfg.propagation.ortho_interval = 10

cfg.pop_control.resample = True
cfg.pop_control.freq = 10  # population control every 10 propagation steps

cfg.log.enabled = True
cfg.log.equil_interval = 0
cfg.log.equil_n_print = 5
cfg.log.block_interval = 1
cfg.log.pop_control_stats = False

e, err = afqmc_energy(hamil, trial=trial, cfg=cfg)
print("E =", float(e), "+/-", float(err))
