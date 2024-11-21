# Hybrid AFQMC

This is the repo of corresponding code for the manuscript:
> Chen, Y., Zhang, L., E, W. & Car, R. (2022). Hybrid Auxiliary Field Quantum Monte Carlo for Molecular Systems. arXiv preprint [arXiv:2211.10824](https://arxiv.org/pdf/2211.10824.pdf).

## Installation
Install the required packages using
```bash
pip install -r requirements.txt 
```
If you have a GPU available (highly recommended for fast training), then you can install JAX with CUDA support, using e.g.:
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

> **Note:** The `jaxlib` version must correspond to your existing CUDA installation. Refer to the [JAX documentation](https://github.com/jax-ml/jax#installation) for detailed instructions.

The [`hafqmc`](./hafqmc/) folder contains all the code and can be used directly as a package. Ensure that it is added to your `PYTHONPATH` to use it in your scripts.

---

## Usage

HAFQMC uses the `ConfigDict` from [ml_collections](https://github.com/google/ml_collections) to configure the system. 

The [`examples`](./examples/) folder contains several example configurations and scripts demonstrated in the manuscript. You can run them directly using:
```bash
python run.py
```

If using `run_fprestart.py`, ensure that you have performed an optimization first. Rename the resulting `checkpoint.pkl` to `oldstates.pkl` before running the script.