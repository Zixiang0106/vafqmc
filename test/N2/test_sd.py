from hafqmc.afqmc import AFQMCConfig, afqmc_energy_from_pickle

cfg = AFQMCConfig.single_det_example()

cfg.pop_control.freq = 5
cfg.propagation.dt = 0.005
cfg.propagation.n_walkers = 400
cfg.propagation.n_blocks = 200
cfg.propagation.n_block_steps = 50
cfg.log.block_freq = 1
cfg.propagation.force_cap = 0.0
e, err = afqmc_energy_from_pickle("hamiltonian.pkl", cfg=cfg)
print("E =", float(e), "+/-", float(err))
