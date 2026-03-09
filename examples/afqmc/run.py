import hafqmc.train
import hafqmc.config

cfg = hafqmc.config.example()

#cfg.molecule = dict(atom="C 0 0 0", basis='aug-ccpvtz', spin=2, unit='B')
cfg.restart.hamiltonian = "hamiltonian.pkl"

cfg.ansatz.propagators[0].max_nhs = 100
cfg.ansatz.propagators[0].aux_network = None
cfg.ansatz.propagators[0].init_tsteps = [0.01] * 1
cfg.ansatz.propagators[0].sqrt_tsvpar = True
cfg.ansatz.propagators[0].init_random = 0.1
cfg.ansatz.propagators[0].hermite_ops = False
cfg.ansatz.propagators[0].mf_subtract = False
cfg.ansatz.propagators[0].spin_mixing = True
cfg.optim.optimizer = "adabelief"
cfg.optim.grad_clip = 1.
cfg.optim.iteration = 5000
cfg.optim.lr.start = 3e-4
cfg.optim.lr.delay = 5e3
cfg.optim.lr.decay = 1.
cfg.sample.size = 8000
cfg.sample.batch = 800
cfg.sample.sampler = {"name": "hmc", "dt": 0.1, "length": 1.}
cfg.sample.burn_in = 100

cfg.loss.sign_factor = 1
cfg.loss.sign_target = 0.7
# cfg.loss.std_factor = 0.1
# cfg.loss.std_target = 100.

cfg.seed = 1
cfg.log.level = "info"

if __name__ == "__main__": 
    hafqmc.train.train(cfg)