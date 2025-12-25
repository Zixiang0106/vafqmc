import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass, field
from typing import Tuple
from .walkers import Walkers 

@jit
def gf(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the greens function G = A^* (B^T A^*)^{-1} B^T
    """
    ovlp = B.T @ A.conj()
    G = A.conj() @ jnp.linalg.inv(ovlp) @ B.T
    return G



@dataclass
class SDTrial:
    """
    Single Slater determinant trial wave function.
    This object holds the state of the trial wavefunction.
    """
    # Core wavefunction arrays
    coeff: jnp.ndarray
    psi: jnp.ndarray
    G: jnp.ndarray

    # Half-rotated integrals, initialized as None
    rh1: jnp.ndarray = field(default=None, repr=False)
    rchol: jnp.ndarray = field(default=None, repr=False)

def create_trial(wavefunction: jnp.ndarray, nelec0: int) -> SDTrial:
    """Factory function to create an SDTrial object."""
    psi = wavefunction[:, :nelec0]
    G = gf(psi, psi)
    return SDTrial(coeff=wavefunction, psi=psi, G=G)

@jit
def half_rotate(trial: SDTrial, hamiltonian) -> SDTrial:
    """
    Computes and returns a NEW trial object with half-rotated integrals.
    """
    rh1 = trial.psi.conj().T @ hamiltonian.h1e
    rchol = jnp.einsum("ij,aik->ajk", trial.psi.conj(), hamiltonian.chol)
    # Return a new instance with updated fields
    return trial.__class__(**{**trial.__dict__, "rh1": rh1, "rchol": rchol})

@jit
def calc_overlap(trial: SDTrial, walkers: Walkers) -> jnp.ndarray:
    """Calculates <Psi_T|phi_w> for all walkers."""
    return jnp.einsum("ij,aik->ajk", trial.psi.conj(), walkers.walker_states)

@jit
def get_ghalf(trial: SDTrial, walkers: Walkers) -> jnp.ndarray:
    """Calculates the half-rotated Green's function Theta."""
    overlap = calc_overlap(trial, walkers)
    overlap_inv = jnp.linalg.inv(overlap)
    Ghalf = jnp.einsum("aij,ajk->aik", walkers.walker_states, overlap_inv)
    return Ghalf

@jit
def calc_force_bias(trial: SDTrial, walkers: Walkers) -> jnp.ndarray:
    """Calculates the force bias."""
    Ghalf = get_ghalf(trial, walkers)
    vbias = 2.0 * jnp.einsum("pij,aji->ap", trial.rchol, Ghalf)
    return vbias