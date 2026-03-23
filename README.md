# Hybrid AFQMC

This repository contains code related to:
- VAFQMC paper: Chen, Y.; Zhang, L.; E, W.; Car, R.  
  Hybrid Auxiliary Field Quantum Monte Carlo for Molecular Systems.  
  [arXiv:2211.10824](https://arxiv.org/abs/2211.10824)
- AFQMC paper: Xiao, Z.-Y.; Lu, Z.; Chen, Y.; Xiang, T.; Zhang, S.  
  Implementing advanced trial wave functions in fermion quantum Monte Carlo via stochastic sampling.  
  [arXiv:2505.18519](https://arxiv.org/abs/2505.18519)

## Installation

Install the package:
```bash
pip install -e .
```


## What Is Included

- `hafqmc.train`: VAFQMC trial training/optimization.
- `hafqmc.afqmc`: AFQMC energy evaluation with stochastic VAFQMC trial (`trial_type="stochastic"`).

All runtime configs use `ml_collections.ConfigDict`.

## Quick Start

### 1) Train a VAFQMC Ansatz (single GPU)

```bash
cd examples/afqmc
python run.py
```

This produces/uses files such as `checkpoint.pkl` and `hparams.yml` for AFQMC trial construction.

### 2) Run AFQMC with Stochastic (VAFQMC) Trial

```bash
cd examples/afqmc
python run_afqmc.py
```

The script loads:
- `hamiltonian.pkl`
- `checkpoint.pkl`
- `hparams.yml`

and runs AFQMC with `AFQMCConfig.stochastic_example()`.

## AFQMC Core Parameters (Stochastic Trial)

Tune parameters in `examples/afqmc/run_afqmc.py` (or your own script using `AFQMCConfig.stochastic_example()`).

`propagation`:
- `dt`: imaginary-time step.
- `n_walkers`: walker count.
- `n_eq_steps`: equilibration steps before production.
- `n_blocks`: number of production blocks.
- `n_block_steps`: propagation steps per block.
- `n_ene_measurements`: number of energy measurements within each block.
- `ortho_freq`: walker orthogonalization frequency.

`pop_control`:
- `resample`: enable/disable population resampling.
- `freq`: population-control frequency in propagation steps.
- `min_weight`, `max_weight`: walker weight clipping window.
- `init_noise`: random noise added to initial walker weights.

`stochastic_trial`:
- `checkpoint`, `hparams_path`: trained VAFQMC trial files.
- `n_samples`: number of left-trial samples per walker.
- `burn_in`: sampler burn-in for runtime trial sampling.
- `sample_update_steps`: trial sampler updates between AFQMC uses.
- `n_measure_samples`: number of trial samples used in measurement.
- `sampler.name`, `sampler.dt`, `sampler.length`: trial sampler setup.
- `local_energy_chunk_size`: walker-axis chunk size for local-energy eval (`0` disables chunking).
- `init_walkers_from_trial`: initialize AFQMC walkers from trial sampler.
- `init_walkers_burn_in`: stage-1 burn-in steps for walker initialization.
- `init_walkers_chains_per_walker`: stage-1 chains per walker (`0` means auto=`n_samples`).
- `init_walkers_infer_steps`: optional trial energy inference steps after stage-1 burn-in.

`log`:
- `enabled`: global AFQMC logging switch.
- `equil_freq`: equilibration print frequency.
- `block_freq`: production block print frequency.
- `pop_control_stats`: print population-control min/max/mean statistics.

`output`:
- `write_raw`, `raw_path`: write per-block raw data.
- `write_hparams`, `hparams_path`: dump AFQMC config for reproducibility.
- `visualization.enabled`: live/update plot generation.
- `visualization.refresh_every`: plot update interval (blocks).
- `visualization.show`: interactive plot display.
- `visualization.save_path`: plot output path.

## Build AFQMC Hamiltonian Pickle from PySCF

```python
from pyscf import gto, scf
from hafqmc.afqmc import build_hamiltonian_pickle

mol = gto.M(
    atom="N 0 0 0; N 3.6 0 0",
    basis="ccpvdz",
    unit="B",
    spin=0,
)
mf = scf.UHF(mol).run()
build_hamiltonian_pickle(mf, "hamiltonian.pkl", chol_cut=1e-6)
```

## Multi-GPU Notes

Example files are provided under `examples/multi_gpu/`:
- `run.py`
- `job.sh`


## More Examples

See `examples/` for additional systems and workflows:
- `examples/N2_d3.6/`
- `examples/N2_d3.6_nn/`
- `examples/cbd_ts/`
- `examples/benzene/`

If you use `run-fprestart.py`, run optimization first and provide the expected checkpoint files.
