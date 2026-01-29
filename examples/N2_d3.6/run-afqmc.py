import hafqmc.afqmc
import hafqmc.config

cfg = hafqmc.config.example()

cfg.restart.hamiltonian = "./hamiltonian.pkl"
cfg.restart.params = "./checkpoints/ckpt_1000.pkl"

# AFQMC settings
cfg.afqmc.dt = 0.01
cfg.afqmc.steps = 2000
cfg.afqmc.walkers = 200
cfg.afqmc.force_cap = 1.5
cfg.afqmc.force_background = "hf"
cfg.afqmc.ET = -1.159
cfg.afqmc.trial_mode = "hf"
cfg.afqmc.stabilize_step = 5
cfg.afqmc.expm_option = ["scan", 6, 1]
cfg.afqmc.phase_metro = True
cfg.afqmc.pop_control = {"enabled": True, "freq": 5, "cap": 0.2}
cfg.afqmc.et_update = {"enabled": True, "gamma": 0.1}
cfg.afqmc.log_burnin_energy = True
cfg.afqmc.trial_config_path = "hparams.yml"

# Block + ICF control
cfg.afqmc.blocking.enabled = True
cfg.afqmc.blocking.writes = 50
cfg.afqmc.blocking.measures = 2
cfg.afqmc.blocking.skip_steps = 5

cfg.afqmc.icf.enabled = True
cfg.afqmc.icf.thermal_steps = 200
cfg.afqmc.icf.et_adjust_step = 10
cfg.afqmc.icf.et_adjust_max = 100
cfg.afqmc.icf.et_bg_estimate_step = 10
cfg.afqmc.icf.et_bg_estimate_max = 100

cfg.afqmc.hmc.dt = 0.1
cfg.afqmc.hmc.length = 1.0
cfg.afqmc.hmc.n_chains = 20
cfg.afqmc.hmc.sweeps = 1
cfg.afqmc.hmc.burn_in = 20

cfg.log.stat_freq = 20
cfg.log.level = "info"

if __name__ == "__main__":
    hafqmc.afqmc.run(cfg)
