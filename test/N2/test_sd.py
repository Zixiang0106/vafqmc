from hafqmc.afqmc import AFQMCConfig, afqmc_energy_from_pickle

cfg = AFQMCConfig.single_det_example()

cfg.pop_control.freq = 5
cfg.propagation.dt = 0.005
cfg.propagation.n_walkers = 400
cfg.propagation.n_blocks = 200
cfg.propagation.n_sr_blocks = 2
cfg.propagation.n_prop_steps = 50
cfg.propagation.log_interval = 1                       
cfg.propagation.force_cap = 0.0
e, err = afqmc_energy_from_pickle("hamiltonian.pkl", cfg=cfg)
print("E =", float(e), "+/-", float(err))
