from hafqmc.afqmc import AFQMCConfig, afqmc_energy_from_pickle

cfg = AFQMCConfig.single_det_example()

cfg.propagation.dt = 0.01
cfg.propagation.n_walkers = 200
cfg.propagation.n_blocks = 100
cfg.propagation.n_block_steps = 50
cfg.log.block_freq = 1

e, err = afqmc_energy_from_pickle("hamiltonian.pkl", cfg=cfg)
print("E =", float(e), "+/-", float(err))
