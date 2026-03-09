from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "flax",
    "optax",
    "pyscf",
    "ml-collections",
    "tensorboardX",
    "jax",
    "jaxlib",
    "numpy",
    "scipy",
    "pyyaml",
    "matplotlib",
    "blackjax",
    "h5py",
]

setup(
    name="hafqmc",
    version="0.1",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    description="A package to perform Hybrid AFQMC calculations",
    python_requires=">=3.9",
)
