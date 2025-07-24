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
    stblz = paxis.all_max(x) 
    exp = jnp.exp(x - stblz)
    if normalize:
        assert normalize.lower() in ("sum", "mean"), "invalid normalize option"
        reducer = getattr(paxis, f"all_{normalize.lower()}")
        total = reducer(lax.stop_gradient(exp))
        exp /= total
        stblz += jnp.log(total)
    return exp, stblz


def make_ovlp_local(hamil: Hamiltonian, braket: BraKet):
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
    slov_fn = hamil.calc_slov

    def eval_ovlp(params_alpha, params_beta, fields):
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
        (bra_alpha, bra_alpha_lw), (ket_alpha, ket_alpha_lw) = braket.apply(params_alpha, fields)
        (bra_beta, bra_beta_lw), (ket_beta, ket_beta_lw) = braket.apply(params_beta, fields)
        Sa, logov_a = slov_fn(bra_alpha, ket_alpha)
        logD_a = logov_a + bra_alpha_lw + ket_alpha_lw
        Sb, logov_b = slov_fn(bra_beta, ket_beta)
        logD_b = logov_b + bra_beta_lw + ket_beta_lw
        Sab, logov_ab = slov_fn(bra_alpha, ket_beta)
        Sba, logov_ba = slov_fn(bra_beta, ket_alpha)
        logR_ab = logov_ab + bra_alpha_lw + ket_beta_lw - logov_a - bra_alpha_lw - ket_alpha_lw
        logR_ba = logov_ba + bra_beta_lw + ket_alpha_lw - logov_b - bra_beta_lw - ket_beta_lw
        return Sa, Sb, logD_a, logD_b, logR_ab, logR_ba, Sab, Sba
        #eloc = eloc_fn(bra, ket)
        #sign, logov = slov_fn(bra, ket)
        #return eloc, sign, logov + bra_lw + ket_lw

    return eval_ovlp

def make_ovlp_total(hamil: Hamiltonian, braket: BraKet,
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
    ovlp_local = make_ovlp_local(hamil, braket)
    batch_ovlp = jax.vmap(ovlp_local, in_axes=(None, None, 0))

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
            n_samples = _f0.shape[0]
            batch = min(n_samples, default_batch)
            n_loops = n_samples // batch
            new_shape = (n_loops, batch) + _f0.shape[1:]
            fields = tree_map(lambda x: x.reshape(*new_shape), fields)
            if logsw is not None:
                logsw = logsw.reshape(n_loops, batch)
        return fields, logsw
    def calc_statistics(Sa, Sb, R_ab, R_ba, logD_a, logD_b, logsw_a, logsw_b):
        logsw_a = logsw_a if logsw_a is not None else logD_a
        logsw_b = logsw_b if logsw_b is not None else logD_b
        logsw_a = lax.stop_gradient(logsw_a)
        logsw_b = lax.stop_gradient(logsw_b)
        rel_wa, lshifta = exp_shifted(logD_a - logsw_a, normalize="mean")
        rel_wb, lshiftb = exp_shifted(logD_b - logsw_b, normalize="mean")
        Numa = paxis.all_mean((Sa * R_ab) * rel_wa)
        Numb = paxis.all_mean((Sb * R_ba) * rel_wb)
        Dema = paxis.all_mean(Sa * rel_wa)
        Demb = paxis.all_mean(Sb * rel_wb)
        ovlp = ((Numa*Numb)/(Dema*Demb)).real
        #ovlp = Numa.real*Numb.real/(Dema.real*Demb.real)
        aux_data = {"ovlp": ovlp,
                    "Numa": Numa,
                    "Numb": Numb,
                    "Dema": Dema,
                    "Demb": Demb,
                    "log_shift_a": lshifta,
                    "log_shift_b": lshiftb}
        return ovlp, aux_data
    def eval_ovlp(params_alpha, params_beta, data_alpha, data_beta):
        data_alpha = check_shape(data_alpha)
        data_beta = check_shape(data_beta)
        fields_a, logsw_a = data_alpha
        fields_b, logsw_b = data_beta
        ovlp_fn = partial(batch_ovlp, params_alpha, params_beta)
        if jax.tree_util.tree_leaves(fields_a)[0].shape[0] > 1:
            ovlp_fn = jax.checkpoint(ovlp_fn, prevent_cse=False)
        Sa, _, logD_a, _, logR_ab, _, Sab, _ = lax.map(ovlp_fn, fields_a)
        _, Sb, _, logD_b, _, logR_ba, _, Sba = lax.map(ovlp_fn, fields_b)
        R_ab = jnp.exp(logR_ab) * Sab / Sa
        R_ba = jnp.exp(logR_ba) * Sba / Sb
        ovlp, aux_data = calc_statistics(Sa, Sb, R_ab, R_ba, logD_a, logD_b, logsw_a, logsw_b)
        return ovlp, aux_data
    return eval_ovlp
