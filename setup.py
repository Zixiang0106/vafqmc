from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "jax==0.4.38",
    "jaxlib==0.4.38",
    "flax==0.10.1",
    "optax==0.2.3",
    "scipy==1.15.3",
    "numpy>=2.0",
    "pyscf>=2.11",
    "ml-collections>=1.1",
    "tensorboardX>=2.6",
    "pyyaml>=6.0",
    "matplotlib>=3.8",
    "h5py>=3.10",
]

setup(
    name="hafqmc",
    version="0.1",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    description="A package to perform Hybrid AFQMC calculations",
    python_requires=">=3.9",
)
