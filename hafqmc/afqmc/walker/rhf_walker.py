import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax.numpy import ndarray

@dataclass
class Walkers:
    walker_states: ndarray  # determinants, shape: (n_walkers, n_basis, n_elec)
    walker_weights: ndarray # weights, shape: (n_walkers,)

    @property
    def nwalkers(self) -> int:
        return self.walker_weights.shape[0]
    
    @property
    def total_weight(self) -> float:
        return jnp.sum(self.walker_weights)


def initialize_walkers(trial, nwalkers: int) -> Walkers:
    """Initialize the walkers with the trial wave function"""
    assert trial.psi is not None
    walker_states = jnp.stack([trial.psi] * nwalkers).astype(jnp.complex128)
    walker_weights = jnp.array([1.0] * nwalkers, dtype=jnp.float64)
    
    return Walkers(walker_states=walker_states, walker_weights=walker_weights)


def reorthogonalize(walkers: Walkers) -> Walkers:
    orthowalkers, _ = jnp.linalg.qr(walkers.walker_states)
    return Walkers(walker_states=orthowalkers, walker_weights=walkers.walker_weights)


def stochastic_reconfiguration(walkers: Walkers, rng_key: ndarray) -> Walkers:
    
    key, subkey = jax.random.split(rng_key)
    walker_weights_rescaled = walkers.walker_weights / walkers.total_weight * walkers.nwalkers
    cumulative_weights = jnp.cumsum(walker_weights_rescaled)
    total_weight = cumulative_weights[-1]
    new_walker_weights = jnp.ones(walkers.nwalkers, dtype=jnp.float64) * (total_weight / walkers.nwalkers)
    zeta = jax.random.uniform(subkey)
    z = total_weight * (jnp.arange(walkers.nwalkers) + zeta) / walkers.nwalkers
    indices = jnp.searchsorted(cumulative_weights, z)
    new_walker_states = walkers.walker_states[indices]
    return Walkers(walker_states=new_walker_states, walker_weights=new_walker_weights)