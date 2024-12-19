import jax
from jax import lax
from jax import numpy as jnp
from functools import partial

from .utils import paxis, just_grad, tree_map
from .hamiltonian import Hamiltonian
from .ansatz import BraKet


def exp_shifted(x, normalize=None):
    # TODO: this all_max is actually all_mean in training
    # TODO: and the all_mean is important in stablizing the gradient
    # TODO: it is effectively a centering of gradients
    # TODO: fix this by making it explicit in eval_total
    # TODO: fix the std penalty
    stblz = paxis.all_max(x) 
    exp = jnp.exp(x - stblz)
    if normalize:
        assert normalize.lower() in ("sum", "mean"), "invalid normalize option"
        reducer = getattr(paxis, f"all_{normalize.lower()}")
        total = reducer(lax.stop_gradient(exp))
        exp /= total
        stblz += jnp.log(total)
    return exp, stblz


def make_eval_local(hamil: Hamiltonian, braket: BraKet):
    """Create a function that evaluates local energy, sign and log of the overlap.

    Args:
        hamil (Hamiltonian): 
            the hamiltonian of the system contains 1- and 2-body integrals.
        braket (Braket): 
            the braket ansatz that generate two Slater determinant (bra and ket)
            and corresponding weights from aux fields.

    Returns:
        eval_local (callable): 
            a function that takes parameters of the propagator and the 
            field configurations for bra and ket (shape: `braket.fields_shape()`)
            and returns the local energy, sign and overlap from the bra and ket.
    """
    eloc_fn = hamil.local_energy
    slov_fn = hamil.calc_slov

    def eval_local(params, fields):
        r"""evaluate the local energy, sign and log-overlap of the bra and ket.

        Args:
            params (dict): 
                the parameter of the propagator ansatz (as a flax linen model)
            fields (array): 
                the field configurations (shape: `braket.fields_shape()`) for both bra and ket

        Returns:
            eloc (float): 
                the local energy :math:`\frac{ <\Phi(\sigma)|H|\Psi(\sigma)> }{ <\Phi(\sigma)|\Psi(\sigma)> }`
            sign (float): 
                the sign of the overlap :math:`\frac{ <\Phi(\sigma)|\Psi(\sigma)> }{ |<\Phi(\sigma)|\Psi(\sigma)>| }`
            logov (float): 
                the log of absolute value of the overlap :math:`\log{ |<\Phi(\sigma)|\Psi(\sigma)>| }`
        """
        (bra, bra_lw), (ket, ket_lw) = braket.apply(params, fields)
        eloc = eloc_fn(bra, ket)
        sign, logov = slov_fn(bra, ket)
        return eloc, sign, logov + bra_lw + ket_lw

    return eval_local


def make_eval_total(hamil: Hamiltonian, braket: BraKet,
                    default_batch: int = 100, calc_stds: bool = False):
    """Create a function that evaluates the total energy from a batch of field configurations.

    Args:
        hamil (Hamiltonian): 
            the hamiltonian of the system contains 1- and 2-body integrals.
        braket (Braket): 
            the braket ansatz that generate two Slater determinant (bra and ket)
            and corresponding weights from aux fields.
        default_batch (int, optional): 
            the batch size to use if there is no pre-spiltted batch in the data.
        calc_stds (bool, optional):
            whether to evaluate the standard deviation of `exp_es` and `exp_s`.

    Returns:
        eval_total (callable): 
            a function that takes parameters and (batched) data that contains 
            field configurations (shape: `(n_loop x) n_batch x braket.fields_shape()`)
            and the corresponding (unnormalized) log density that they are sampled from,
            and returns the estimated total energy and auxiliary estimations. 
    """

    eval_local = make_eval_local(hamil, braket)
    batch_eval = jax.vmap(eval_local, in_axes=(None, 0))

    def check_shape(data):
        fields, logsw = data
        if isinstance(fields, jnp.ndarray):
            fields = (fields,)
        fshape = braket.fields_shape(len(fields) if braket.trial is None
                                     else tuple(map(len, fields)))
        # _f0 and _fs0 are just for checking the shape
        _f0 = jax.tree_util.tree_leaves(fields)[0]
        _fs0 = jax.tree_util.tree_leaves(fshape)[0]
        if _f0.ndim != _fs0.size + 2:
            batch = min(_f0.size // _fs0.prod(), default_batch)
            fields = tree_map(lambda x,s: x.reshape(-1, batch, *s), fields, fshape)
            if logsw is not None:
                logsw = logsw.reshape(-1, batch)
        return fields, logsw

    def calc_statistics(eloc, sign, logov, logsw, ebar=None):
        logsw = logsw if logsw is not None else logov
        logsw = lax.stop_gradient(logsw)
        rel_w, lshift = exp_shifted(logov - logsw, normalize="mean")
        exp_es = paxis.all_mean((eloc * sign) * rel_w)
        exp_s = paxis.all_mean(sign * rel_w)
        etot = (exp_es / exp_s).real
        aux_data = {"e_tot": etot, 
                    "exp_es": exp_es.real, 
                    "exp_s": exp_s.real,
                    "log_shift": lshift}
        if calc_stds:
            tot_w = paxis.all_mean(rel_w) # should be just 1, but provide correct gradient
            var_es = paxis.all_mean(jnp.abs(eloc*sign - exp_es/tot_w)**2 * rel_w) / tot_w
            var_s = paxis.all_mean(jnp.abs(sign - exp_s/tot_w)**2 * rel_w) / tot_w
            aux_data.update(std_es=jnp.sqrt(var_es), std_s=jnp.sqrt(var_s))
        if ebar is not None:
            aux_data["e_bar"] = ebar
            ediff = lax.stop_gradient(etot - ebar).real
            etot += just_grad(ediff * exp_s.real / lax.stop_gradient(exp_s.real))
        return etot, aux_data
            
    def eval_total(params, data, ebar=None):
        r"""evaluate the total energy and the auxiliary estimations from batched data.

        Args:
            params (dict): 
                the parameter of the propagator ansatz (as a flax linen model)
            data (tuple of array): 
                a tuple like (fields, logsw), for field configurations and corresponding log density
                that they are sampled from (for important sampling purpose). The fields are of shape
                `(n_loop x) n_batch x braket.fields_shape()` and logsw of shape `(n_loop x) n_batch`.
                The function would loop for `n_loop` times with each time eval a batch size n_batch.
                If n_loop is not given, calculated from `default_batch` (as the maximum batch size).

        Returns:
            e_tot (float): 
                the estimated total energy :math:`\frac{ <\Phi|H|\Psi> }{ <\Phi|\Psi> }`
            aux_data (tuple): 
                the dict for auxiliary estimated data, by default {`e_tot`, `exp_es`, `exp_s`}.
                If `calc_stds` then will also add {`std_es`, `std_s`} into the dict.
                where `exp_es`, `std_es` are the estimated mean and std of `(eloc * sign).real`,
                and `exp_s`, `std_s` are the estimated mean and std of `sign.real`.
        """
        data = check_shape(data)
        fields, logsw = data
        eval_fn = partial(batch_eval, params)
        if jax.tree_util.tree_leaves(fields)[0].shape[0] > 1:
            eval_fn = jax.checkpoint(eval_fn, prevent_cse=False)
        eloc, sign, logov = lax.map(eval_fn, fields)
        etot, aux_data = calc_statistics(eloc, sign, logov, logsw, ebar)
        return etot, aux_data
            
    return eval_total


#TODO:clean up the code
def make_rdm_local(hamil: Hamiltonian, braket: BraKet):
    """Create a function that evaluates the RDM, sign, and log of the overlap."""
    rdm_fn = hamil.calc_rdm
    slov_fn = hamil.calc_slov

    def rdm_local(params, fields):
        """Evaluate the RDM, sign, and log-overlap of the bra and ket."""
        (bra, bra_lw), (ket, ket_lw) = braket.apply(params, fields)
        rdm = rdm_fn(bra, ket)
        sign, logov = slov_fn(bra, ket)
        return rdm, sign, logov + bra_lw + ket_lw

    return rdm_local


def make_rdm_total(hamil: Hamiltonian, braket: BraKet,
                   default_batch: int = 100, calc_stds: bool = False):
    """Create a function that evaluates the total RDM from a batch of field configurations."""
    rdm_local = make_rdm_local(hamil, braket)
    batch_rdm = jax.vmap(rdm_local, in_axes=(None, 0))

    def check_shape(data):
        fields, logsw = data
        if isinstance(fields, jnp.ndarray):
            fields = (fields,)
        fshape = braket.fields_shape(len(fields) if braket.trial is None
                                     else tuple(map(len, fields)))
        _f0 = jax.tree_util.tree_leaves(fields)[0]
        _fs0 = jax.tree_util.tree_leaves(fshape)[0]
        if _f0.ndim != _fs0.size + 2:
            batch = min(_f0.size // _fs0.prod(), default_batch)
            fields = tree_map(lambda x, s: x.reshape(-1, batch, *s), fields, fshape)
            if logsw is not None:
                logsw = logsw.reshape(-1, batch)
        return fields, logsw
    def calc_statistics(rdm, sign, logov, logsw, ebar=None):
        logsw = logsw if logsw is not None else logov
        logsw = lax.stop_gradient(logsw)
        # Stabilize and normalize weights
        rel_w, lshift = exp_shifted(logov - logsw, normalize="mean")
        # Expand dimensions for broadcasting
        rel_w = rel_w[..., None, None]
        sign = sign[..., None, None]
        # Weighted accumulation of RDM
        weighted_rdm = rdm * sign * rel_w
        # Compute mean across batch dimensions
        batch_axes = (0, 1)  # Batch dimensions are the first two axes
        exp_rdm = jnp.mean(weighted_rdm, axis=batch_axes)  # Reduce batch axes only
        exp_s = jnp.mean(sign * rel_w, axis=batch_axes)    # Reduce batch axes only
        # Normalize RDM by overlap
        rdm_tot = (exp_rdm / exp_s).real  # Normalize without additional dimensions
        # Calculate variance and standard deviation
        rel_w_sum = jnp.sum(rel_w, axis=batch_axes)
        mean_rdm = exp_rdm / exp_s
        var_rdm = (jnp.sum(
            rel_w * (rdm - mean_rdm)**2, axis=batch_axes
        ) / rel_w_sum).real
        std_rdm = jnp.sqrt(var_rdm)

        aux_data = {
            "rdm_tot": rdm_tot,
            "exp_rdm": exp_rdm,
            "exp_s": exp_s,
            "log_shift": lshift,
            "std_rdm": std_rdm,  # Include standard deviation in auxiliary data
        }

        if calc_stds:
            var_s = jnp.mean(jnp.abs(sign - exp_s / rel_w_sum)**2 * rel_w, axis=batch_axes)
            aux_data.update(std_s=jnp.sqrt(var_s))

        return rdm_tot, aux_data

    def rdm_total(params, data, ebar=None):
        """Evaluate the total RDM and auxiliary data from batched field configurations."""
        data = check_shape(data)
        fields, logsw = data
        rdm_fn = partial(batch_rdm, params)

        if jax.tree_util.tree_leaves(fields)[0].shape[0] > 1:
            rdm_fn = jax.checkpoint(rdm_fn, prevent_cse=False)
        # Execute RDM computation
        rdm, sign, logov = lax.map(rdm_fn, fields)
        # Calculate statistics
        rdm_tot, aux_data = calc_statistics(rdm, sign, logov, logsw, ebar)
        return rdm_tot, aux_data
    return rdm_total