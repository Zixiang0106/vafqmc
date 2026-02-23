from hafqmc.afqmc import AFQMCConfig, afqmc_energy_from_pickle

cfg = AFQMCConfig.single_det_example()

cfg.propagation.dt = 0.01
cfg.propagation.n_walkers = 200
cfg.propagation.n_blocks = 100
cfg.propagation.n_prop_steps = 50
cfg.propagation.log_interval = 1

e, err = afqmc_energy_from_pickle("hamiltonian.pkl", cfg=cfg)
print("E =", float(e), "+/-", float(err))
