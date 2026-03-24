from hafqmc.afqmc import AFQMCConfig, afqmc_energy_from_pickle
cfg = AFQMCConfig.stochastic_example()

# ---------------------------------------------------------------------------
# Global
# ---------------------------------------------------------------------------
cfg.seed = 0
cfg.trial_type = "stochastic"

# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------
cfg.propagation.dt = 0.005
cfg.propagation.n_walkers = 100
cfg.propagation.n_block_steps = 50
cfg.propagation.n_ene_measurements = 1
cfg.propagation.n_blocks = 200
cfg.propagation.n_eq_steps = 200
cfg.propagation.ortho_freq = 10

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
cfg.log.enabled = True
cfg.log.block_freq = 1
cfg.log.equil_freq = 40

# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------
cfg.output.write_raw = True
cfg.output.raw_path = "raw.dat"
cfg.output.write_hparams = True
cfg.output.hparams_path = "afqmc_hparams.yml"
cfg.output.visualization.enabled = True
cfg.output.visualization.show = False
cfg.output.visualization.save_path = "eblock_afqmc.png"

# ---------------------------------------------------------------------------
# Population control
# ---------------------------------------------------------------------------
cfg.pop_control.init_noise = 0.0
cfg.pop_control.resample = True
cfg.pop_control.freq = 10
cfg.pop_control.min_weight = 1.0e-8
cfg.pop_control.max_weight = 100.0

# ---------------------------------------------------------------------------
# Stochastic (VAFQMC) trial
# ---------------------------------------------------------------------------
cfg.stochastic_trial.checkpoint = "checkpoint.pkl"
cfg.stochastic_trial.hparams_path = "hparams.yml"
cfg.stochastic_trial.n_samples = 20
cfg.stochastic_trial.burn_in = 100
cfg.stochastic_trial.sample_update_steps = 1
cfg.stochastic_trial.n_measure_samples = 20
cfg.stochastic_trial.sampler.name = "hmc"
cfg.stochastic_trial.sampler.dt = 0.1
cfg.stochastic_trial.sampler.length = 1.0
cfg.stochastic_trial.local_energy_chunk_size = 0
cfg.stochastic_trial.init_walkers_from_trial = True
cfg.stochastic_trial.init_walkers_infer_steps = 20
cfg.stochastic_trial.init_walkers_burn_in = 200

e, err = afqmc_energy_from_pickle("hamiltonian.pkl", cfg=cfg)
print("E =", float(e), "+/-", float(err))
