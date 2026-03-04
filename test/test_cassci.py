from pyscf import gto, mcscf, scf

from hafqmc.afqmc import AFQMCConfig, CASSCITrial, afqmc_energy
from hafqmc.hamiltonian import Hamiltonian


mol = gto.M(
    atom="H 0 0 0; H 0 0 1.4",
    basis="sto-3g",
    unit="B",
    spin=0,
    verbose=4,
)

mf = scf.RHF(mol).run()
hamil = Hamiltonian.from_pyscf(mf, chol_cut=1e-6, orth_ao=None, full_eri=False)

casci = mcscf.CASCI(mf, ncas=2, nelecas=2)
casci.kernel()

trial = CASSCITrial.from_pyscf_casci(
    casci,
    n_det=5,
    coeff_cutoff=1e-10,
    normalize_coeffs=True,
    init_mode="sample_coeff",
    orth_ao=None,
)

cfg = AFQMCConfig.single_det_example()
cfg.seed = 0
cfg.propagation.dt = 0.01
cfg.propagation.n_walkers = 100
cfg.propagation.n_eq_steps = 20
cfg.propagation.n_blocks = 50
cfg.propagation.n_prop_steps = 20
cfg.propagation.log_interval = 1
cfg.pop_control.freq = 5
cfg.pop_control.resample = True

e, err = afqmc_energy(hamil, trial=trial, cfg=cfg)
print("E =", float(e), "+/-", float(err))
