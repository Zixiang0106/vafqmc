import hafqmc
from hafqmc.molecule import build_mf
from hafqmc.afqmc import build_hamiltonian_pickle, AFQMCConfig, afqmc_energy_from_pickle

mf = build_mf(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0)
build_hamiltonian_pickle(mf, "hamiltonian.pkl", chol_cut=1e-6)

cfg = AFQMCConfig(dt=0.01, n_walkers=200, n_blocks=100, n_prop_steps=50, log_interval=1)
e, err = afqmc_energy_from_pickle("hamiltonian.pkl", cfg=cfg)
