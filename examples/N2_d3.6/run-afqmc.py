import hafqmc.afqmc
import hafqmc.config

cfg = hafqmc.config.example()

cfg.restart.hamiltonian = "hamiltonian.pkl"
cfg.restart.params = "oldstates.pkl"

# AFQMC settings
cfg.afqmc.dt = 0.01
cfg.afqmc.steps = 1000
cfg.afqmc.walkers = 200
cfg.afqmc.force_cap = 1.5
cfg.afqmc.force_background = "hf"
cfg.afqmc.ET = -10.0
cfg.afqmc.stabilize_step = 5
cfg.afqmc.expm_option = ["scan", 6, 1]
cfg.afqmc.phase_metro = True
cfg.afqmc.pop_control = {"enabled": True, "freq": 5, "cap": 0.2}
cfg.afqmc.et_update = {"enabled": True, "gamma": 0.1}
cfg.afqmc.log_burnin_energy = True

cfg.afqmc.hmc.dt = 0.1
cfg.afqmc.hmc.length = 1.0
cfg.afqmc.hmc.n_chains = 20
cfg.afqmc.hmc.sweeps = 1
cfg.afqmc.hmc.burn_in = 10

cfg.log.level = "info"

if __name__ == "__main__":
    hafqmc.afqmc.run(cfg)
