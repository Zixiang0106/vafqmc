from hafqmc.afqmc import AFQMCConfig, afqmc_energy_from_pickle

cfg = AFQMCConfig.stochastic_example()
cfg.seed = 0

cfg.propagation.dt = 0.01
cfg.propagation.n_walkers = 200
cfg.propagation.n_eq_steps = 10
cfg.propagation.n_blocks = 100
cfg.propagation.n_prop_steps = 50

cfg.trial_type = "stochastic"
cfg.stochastic_trial.checkpoint = "checkpoints/checkpoint.pkl"
cfg.stochastic_trial.hparams_path = "hparams.yml"
cfg.stochastic_trial.n_samples = 20
cfg.stochastic_trial.burn_in = 1000
cfg.stochastic_trial.n_measure_samples = 20
cfg.stochastic_trial.sampler.name = "hmc"
cfg.stochastic_trial.sampler.dt = 0.1
cfg.stochastic_trial.sampler.length = 1.0

e, err = afqmc_energy_from_pickle("hamiltonian.pkl", cfg=cfg)
print("E =", float(e), "+/-", float(err))