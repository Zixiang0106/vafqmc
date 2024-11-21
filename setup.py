from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
        'flax',
        'optax',
        'pyscf',
        'ml-collections',
        'tensorboardX',
        'jax',          
        'jaxlib',       
    ]
setup(
    name='hafqmc',              
    version='0.1',             
    packages=find_packages(), 
    install_requires=REQUIRED_PACKAGES,
    description='A package to prtform Hybrid AFQMC calculations',
    python_requires='>=3.9',
)
