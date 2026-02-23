from hafqmc.afqmc import AFQMCConfig, afqmc_energy_from_checkpoint

cfg = AFQMCConfig(
    dt=0.01,
    n_walkers=100,
    n_eq_steps=10,
    n_blocks=100,
    n_prop_steps=8,
    n_measure_samples=20,  # per-block trial-only resampling measurements
)
e, err = afqmc_energy_from_checkpoint(
    "hamiltonian.pkl",
    "checkpoints/checkpoint.pkl",
    hparams_path="hparams.yml",
    cfg=cfg,
    n_samples=20,                     # HMC chain/pool size
    n_walkers=100,
    burn_in=200,
    sampler_name="hmc",
    sampler_kwargs={"dt": 0.1, "length": 1.0},
    seed=0,
)
