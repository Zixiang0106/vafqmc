import jax
from jax import lax
from jax import numpy as jnp
from jax import scipy as jsp
from jax.tree_util import tree_map
from flax import linen as nn
from ml_collections import ConfigDict
from typing import Dict, Sequence, Union, Callable, Any, Optional
from functools import partial, reduce
import dataclasses
import pickle
import time


_t_real = jnp.float64
_t_cplx = jnp.complex128


Array = jnp.ndarray
PyTree = Any


def compose(*funcs):
    def c2(f, g):
        return lambda *a, **kw: f(g(*a, **kw))
    return reduce(c2, funcs)


def just_grad(x):
    return x - lax.stop_gradient(x)


def _T(x): 
    return jnp.swapaxes(x, -1, -2)

def _H(x): 
    return jnp.conj(_T(x))


def symmetrize(x): 
    return (x + _H(x)) / 2


def cmult(x1, x2):
    return ((x1.real * x2.real - x1.imag * x2.imag) 
        + 1j * (x1.imag * x2.real + x1.real * x2.imag))


def symrange(nmax, dtype=None):
    return jnp.arange(-nmax, nmax+1, dtype=dtype)


def scatter(value, mask):
    if value.dtype.kind == "c":
        dtype = value.real.dtype
        return (jnp.zeros_like(mask, dtype=dtype).at[mask].set(value.real) 
                + 1j * jnp.zeros_like(mask, dtype=dtype).at[mask].set(value.imag))
    else:
        return jnp.zeros_like(mask, dtype=value.dtype).at[mask].set(value) 
    

@partial(jax.custom_jvp, nondiff_argnums=(1,))
def chol_qr(x, shift=None):
    *_, m, n = x.shape
    a = _H(x) @ x
    if shift is None:
        shift = 1.2e-15 * (m*n + n*(n+1)) * a.trace(0,-1,-2).max()
    r = jsp.linalg.cholesky(a + shift * jnp.eye(n, dtype=x.dtype), lower=False)
    q = lax.linalg.triangular_solve(r, x, left_side=False, lower=False)
    return q, r

@chol_qr.defjvp
def _chol_qr_jvp(shift, primals, tangents):
    x, = primals
    dx, = tangents
    *_, m, n = x.shape
    if m < n:
        raise NotImplementedError("Unimplemented case of QR decomposition derivative")
    q, r = chol_qr(x, shift=shift)
    dx_rinv = lax.linalg.triangular_solve(r, dx)
    qt_dx_rinv = jnp.matmul(_H(q), dx_rinv)
    qt_dx_rinv_lower = jnp.tril(qt_dx_rinv, -1)
    do = qt_dx_rinv_lower - _H(qt_dx_rinv_lower)  # This is skew-symmetric
    # The following correction is necessary for complex inputs
    I = lax.expand_dims(jnp.eye(n, dtype=do.dtype), range(qt_dx_rinv.ndim - 2))
    do = do + I * (qt_dx_rinv - jnp.real(qt_dx_rinv))
    dq = jnp.matmul(q, do - qt_dx_rinv) + dx_rinv
    dr = jnp.matmul(qt_dx_rinv - do, r)
    return (q, r), (dq, dr)


ExpmFnType = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

def make_expm_apply(method="scan", m=6, s=1, matmul_fn=None) -> ExpmFnType:
    # customize matrix multiplication
    matmul = jnp.matmul if matmul_fn is None else matmul_fn
    # the native python loop, slow compiling
    def expm_apply_loop(A, B):
        # n = A.shape[-1]
        # mu = jnp.trace(A, axis1=-1, axis2=-2) / n
        # eta = jnp.expand_dims(jnp.exp(mu), -1)
        # A = A - mu * jnp.identity(n, dtype=A.dtype)
        F = B
        for _ in range(s):
            for n in range(1, m + 1):
                B = matmul(A, B) / (s * n)
                F = F + B
            B = F
        return F # * eta
    # the jax scan version, faster compiling
    def expm_apply_scan(A, B):
        # n = A.shape[-1]
        # mu = jnp.trace(A, axis1=-1, axis2=-2) / n
        # eta = jnp.expand_dims(jnp.exp(mu), -1)
        # A = A - mu * jnp.identity(n, dtype=A.dtype)
        ns = jnp.arange(1., m+1., dtype=A.dtype)
        def _loop_m(B_and_F, n):
            B, F = B_and_F
            B = matmul(A, B) / (s * n)
            return (B, F + B), None
        def _loop_s(B, _):
            (_, B), _ = lax.scan(_loop_m, (B, B), ns)
            return B, None
        B, _ = lax.scan(_loop_s, B, None, s)
        return B # * eta
    # the exact verison, slow execution
    def expm_apply_exact(A, B):
        exp_A = jsp.linalg.expm(A)
        return exp_A @ B
    # the exact verison for A in diag form 
    def expm_apply_diag(A, B):
        if A.ndim==1:
            F = B
            for _ in range(s):
                for n in range(1, m + 1):
                    B = A[:, jnp.newaxis] * B / (s * n)
                    F = F + B
                B = F
            return F
            #############################
            # ns = jnp.arange(1., m+1., dtype=A.dtype)
            # def _loop_m(B_and_F, n):
            #     B, F = B_and_F
            #     # B = matmul(A, B) / (s * n)
            #     B = A[:, jnp.newaxis] * B / (s * n)
            #     # B = block_matrix_multiply(A, B, int(A.shape[0]/4)) / (s * n)
            #     return (B, F + B), None
            # def _loop_s(B, _):
            #     (_, B), _ = lax.scan(_loop_m, (B, B), ns)
            #     return B, None
            # B, _ = lax.scan(_loop_s, B, None, s)
            # return B # * eta
            ##############################
        else:
            # return expm_apply_scan(A, B)
            return expm_apply_loop(A, B)
            # return expm_apply_exact(A, B)
    # choose the function from the method name
    if method == "loop":
        return expm_apply_loop
    if method == "scan":
        return expm_apply_scan
    if method == "diag":
        return expm_apply_diag
    if method == "exact":
        assert matmul_fn is None, 'do not support custom matmul for exact expm'
        return expm_apply_exact
    raise ValueError(f"unknown expm_apply method type: {method}")

def warp_spin_expm(fun_expm: ExpmFnType) -> ExpmFnType:
    def new_expm(A, B):
        if A.shape[-1] == B.shape[-2]:
            return fun_expm(A, B)
        elif A.shape[-1]*2 == B.shape[-2]:
            nao = A.shape[-1]
            nelec = B.shape[-1]
            fB = B.reshape(2, nao, nelec).swapaxes(0,1).reshape(nao, 2*nelec)
            nfB = fun_expm(A, fB)
            nB = nfB.reshape(nao, 2, nelec).swapaxes(0,1).reshape(2*nao, nelec)
            return nB
        else:
            return fun_expm(A, B)
    return new_expm

DEFAULT_EXPM = warp_spin_expm(make_expm_apply("scan", 6, 1))


def make_moving_avg(decay=0.99, early_growth=True):
    def moving_avg(acc, new, i):
        if early_growth:
            iteration_decay = jnp.minimum(decay, (1.0 + i) / (10.0 + i))
        else:
            iteration_decay = decay
        updated_acc = iteration_decay * acc
        updated_acc += (1 - iteration_decay) * new
        return jax.lax.stop_gradient(updated_acc)
    return moving_avg


def ravel_shape(target_shape):
    from jax.flatten_util import ravel_pytree
    tmp = tree_map(jnp.zeros, target_shape)
    flat, unravel_fn = ravel_pytree(tmp)
    return flat.size, unravel_fn


def tree_where(condition, x, y):
    return tree_map(partial(jnp.where, condition), x, y)


def fix_init(key, value, dtype=None, random=0., rnd_additive=False):
    value = jnp.asarray(value, dtype=dtype)
    if random <= 0.:
        return value
    else:
        perturb = jax.random.truncated_normal(
            key, -2, 2, value.shape, _t_real) * random
        if rnd_additive:
            return value + perturb
        else:
            return value * (1 + perturb)


def pack_spin(wfn):
    if not (isinstance(wfn, (tuple, list)) or wfn.ndim >= 3):
        return wfn, wfn.shape[-1]
    w_up, w_dn = wfn
    n_up, n_dn = w_up.shape[-1], w_dn.shape[-1]
    w_packed = jnp.concatenate((w_up, w_dn), -1)
    return w_packed, (n_up, n_dn)

def unpack_spin(wfn, nelec):
    if isinstance(nelec, int):
        return wfn
    n_up, n_dn = nelec
    w_up = wfn[:, :n_up]
    w_dn = wfn[:, n_up : n_up+n_dn]
    return (w_up, w_dn)


def block_spin(a, b, perturb=0.):
    p1 = jnp.eye(a.shape[-2], b.shape[-1]) * perturb
    p2 = jnp.eye(b.shape[-2], a.shape[-1]) * perturb
    return jnp.block([[a, p1],[p2, b]])


def parse_activation(name, **kwargs):
    if not isinstance(name, str):
        return name
    raw_fn = getattr(nn, name)
    return partial(raw_fn, **kwargs)


def parse_bool(keys, inputs):
    if isinstance(keys, str):
        return parse_bool((keys,), inputs)[keys]
    res_dict = {}
    if isinstance(inputs, str) and inputs.lower() in ("all", "true"):
        inputs = True
    if isinstance(inputs, str) and inputs.lower() in ("none", "false"):
        inputs = False
    if isinstance(inputs, bool):
        for key in keys:
            res_dict[key] = inputs
    else:
        for key in keys:
            res_dict[key] = key in inputs
    return res_dict


def ensure_mapping(obj, default_key="name"):
    try:
        return dict(**obj)
    except TypeError:
        return {default_key: obj}


def save_pickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def cfg_to_dict(cfg):
    if not isinstance(cfg, ConfigDict):
        return cfg
    return tree_map(cfg_to_dict, cfg.to_dict())

def cfg_to_yaml(cfg):
    import yaml
    from yaml import representer
    representer.Representer.add_representer(
        dict,
        lambda self, data: self.represent_mapping(
            'tag:yaml.org,2002:map', data, False))
    return yaml.dump(cfg_to_dict(cfg), default_flow_style=None)

def dict_to_cfg(cdict, **kwargs):
    if not isinstance(cdict, (dict, ConfigDict)):
        return cdict
    tree_type = (tuple, list)
    cfg = ConfigDict(cdict, **kwargs)
    for k, v in cfg.items():
        if isinstance(v, ConfigDict):
            cfg[k] = dict_to_cfg(v, **kwargs)
        if type(v) in tree_type:
            cfg[k] = type(v)(dict_to_cfg(vi, **kwargs) for vi in v)
    return cfg
    

class Serial(nn.Module):
    layers : Sequence[nn.Module]
    skip_cxn : bool = True
    actv_fun : Union[str, Callable] = "gelu"

    @nn.compact
    def __call__(self, x):
        actv = parse_activation(self.actv_fun)
        for i, lyr in enumerate(self.layers):
            tmp = lyr(x)
            if i != len(self.layers) - 1:
                tmp = actv(tmp)
            if self.skip_cxn:
                if x.shape[-1] >= tmp.shape[-1]:
                    x = x[...,:tmp.shape[-1]] + tmp
                else:
                    x = tmp.at[...,:x.shape[-1]].add(x)
            else:
                x = tmp
        return x


class Printer:

    def __init__(self, 
                 field_format: Dict[str, Optional[str]], 
                 time_format: Optional[str]=None,
                 **print_kwargs):
        all_format = {**field_format, "time": time_format}
        all_format = {k: v for k, v in all_format.items() if v is not None}
        self.fields = all_format
        self.header = "\t".join(self.fields.keys())
        self.format = "\t".join(f"{{{k}:{v}}}" for k, v in self.fields.items())
        self.kwargs = print_kwargs
        self.tick = time.perf_counter()

    def print_header(self, prefix: str = ""):
        print(prefix+self.header, **self.kwargs)

    def print_fields(self, field_dict: Dict[str, Any], prefix: str = ""):
        output = self.format.format(**field_dict, time=time.perf_counter()-self.tick)
        print(prefix+output, **self.kwargs)

    def reset_timer(self):
        self.tick = time.perf_counter()


def wrap_if_pmap(p_func):

    def p_func_if_pmap(obj, axis_name):
        try:
            jax.core.axis_frame(axis_name)
            return p_func(obj, axis_name)
        except NameError:
            return obj

    return p_func_if_pmap


pmax_if_pmap = wrap_if_pmap(lax.pmax)
pmin_if_pmap = wrap_if_pmap(lax.pmin)
psum_if_pmap = wrap_if_pmap(lax.psum)
pmean_if_pmap = wrap_if_pmap(lax.pmean)


@dataclasses.dataclass(frozen=True)
class PAxis:
    name  : str
    def __post_init__(self):
        for nm, fn in (("vmap", jax.vmap), ("pmap", jax.pmap),
                       ("pmax", pmax_if_pmap), ("pmin", pmin_if_pmap),
                       ("psum", psum_if_pmap), ("pmean", pmean_if_pmap)):
            object.__setattr__(self, nm, partial(fn, axis_name=self.name))
        for nm in ("max", "min", "sum", "mean"):
            jnp_fn = getattr(jnp, nm)
            pax_fn = getattr(self, f"p{nm}")
            all_fn = compose(pax_fn, jnp_fn)
            object.__setattr__(self, f"all_{nm}", all_fn)

PMAP_AXIS_NAME = "_pmap_axis"
paxis = PAxis(PMAP_AXIS_NAME)


# currently slower than vanilla convolution
def fftconvolve(in1, in2, mode='full', axes=None):
    from scipy.signal._signaltools import _init_freq_conv_axes

    def _freq_domain_conv(in1, in2, axes, shape):
        """Convolve `in1` with `in2` in the frequency domain."""
        if not len(axes):
            return in1 * in2
        if (in1.dtype.kind == 'c' or in2.dtype.kind == 'c'):
            fft, ifft = jnp.fft.fftn, jnp.fft.ifftn
        else:
            fft, ifft = jnp.fft.rfftn, jnp.fft.irfftn
        in1_freq = fft(in1, shape, axes=axes)
        in2_freq = fft(in2, shape, axes=axes)
        ret = ifft(in1_freq * in2_freq, shape, axes=axes)
        return ret

    def _apply_conv_mode(ret, s1, s2, mode, axes):
        """Slice result based on the given `mode`."""
        from scipy.signal._signaltools import _centered
        if mode == 'full':
            return ret
        elif mode == 'same':
            return _centered(ret, s1)
        elif mode == 'valid':
            shape_valid = [ret.shape[a] if a not in axes else s1[a] - s2[a] + 1
                                        for a in range(ret.ndim)]
            return _centered(ret, shape_valid)
        else:
            raise ValueError("acceptable mode flags are 'valid', 'same', or 'full'")

    if in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.ndim == in2.ndim == 0:
        return in1 * in2
    elif in1.size == 0 or in2.size == 0:
        return jnp.array([], dtype=in1.dtype)

    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False)
    s1, s2 = in1.shape, in2.shape
    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
                     for i in range(in1.ndim)]
    ret = _freq_domain_conv(in1, in2, axes, shape)
    return _apply_conv_mode(ret, s1, s2, mode, axes)


#copy from jax.scipy.signal._convolve_nd
def rawcorr(in1, in2, mode='full', *, precision=None):
    """same as scipy.signal.correlate but do not do conjugate."""
    from jax._src.numpy.util import _promote_dtypes_inexact
    if mode not in ["full", "same", "valid"]:
        raise ValueError("mode must be one of ['full', 'same', 'valid']")
    if in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 must have the same number of dimensions")
    if in1.size == 0 or in2.size == 0:
        raise ValueError(f"zero-size arrays not supported in convolutions, got shapes {in1.shape} and {in2.shape}.")
    in1, in2 = _promote_dtypes_inexact(in1, in2)

    no_swap = all(s1 >= s2 for s1, s2 in zip(in1.shape, in2.shape))
    swap = all(s1 <= s2 for s1, s2 in zip(in1.shape, in2.shape))
    if not (no_swap or swap):
        raise ValueError("One input must be smaller than the other in every dimension.")

    shape_o = in2.shape
    if not no_swap: # do not swap for same size
        in1, in2 = in2, in1
    shape = in2.shape
    # do not flip in2 here
    # in2 = jnp.flip(in2)

    if mode == 'valid':
        padding = [(0, 0) for s in shape]
    elif mode == 'same':
        padding = [(s - 1 - (s_o - 1) // 2, s - s_o + (s_o - 1) // 2)
                             for (s, s_o) in zip(shape, shape_o)]
    elif mode == 'full':
        padding = [(s - 1, s - 1) for s in shape]

    strides = tuple(1 for s in shape)
    result = lax.conv_general_dilated(in1[None, None], in2[None, None], strides,
                                                                        padding, precision=precision)
    return result[0, 0]


import os
import sys
afqmclab_dir = os.environ.get('AFQMCLAB_DIR')
if afqmclab_dir is None:
    afqmclab_dir = '/mnt/home/zlu10/lib_install/afqmclab'
sys.path.append(afqmclab_dir + "/scripts/pyscf")
from molecule import *
from rhf import *
from uhf import *
from model import *
import numpy as np

import pyscf
from   pyscf  import scf

def writeROHFSD_icf(mol=None, rhf=None, canonic=None, filename=None, noise=0.0):

    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2

    wf = rhf.mo_coeff[:, 0:max(Nup, Ndn)]
    wf = np.dot( canonic.XInv, wf )
    f = open(filename, 'w')

    f.write('{:26.18e} {:26.18e} \n'.format(0.0,0.0))

    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(2*canonic.L,Nup+Ndn))
    for i in range(Nup):
        for j in range(2*canonic.L):
            if j<canonic.L :
                f.write( '{:26.18e} {:26.18e} \n'.format( wf[j,i]+noise*random.random(),0.0 ) )
            else:
                f.write( '{:26.18e} {:26.18e} \n'.format( 0.0,0.0 ) )

    for i in range(Ndn):
        for j in range(2*canonic.L):
            if j<canonic.L :
                f.write( '{:26.18e} {:26.18e} \n'.format( 0.0,0.0 ) )
            else:
                f.write( '{:26.18e} {:26.18e} \n'.format( wf[j-canonic.L,i]+noise*random.random(),0.0 ) )
    f.close()

def writeUHFSD_icf(mol=None, uhf=None, canonic=None, filename=None, noise=0.0):

    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2

    wfUp = np.array(uhf.mo_coeff)[0, :, 0:Nup]
    wfUp = np.dot( canonic.XInv, wfUp )
    wfDn = np.array(uhf.mo_coeff)[1, :, 0:Ndn]
    wfDn = np.dot( canonic.XInv, wfDn )

    f = open(filename, 'w')

    f.write('{:26.18e} {:26.18e} \n'.format(0.0,0.0))

    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(2*canonic.L,Nup+Ndn))
    for i in range(Nup):
        for j in range(2*canonic.L):
            if j<canonic.L :
                f.write( '{:26.18e} {:26.18e} \n'.format( wfUp[j,i]+noise*random.random(),0.0 ) )
            else:
                f.write( '{:26.18e} {:26.18e} \n'.format( 0.0,0.0 ) )

    for i in range(Ndn):
        for j in range(2*canonic.L):
            if j<canonic.L :
                f.write( '{:26.18e} {:26.18e} \n'.format( 0.0,0.0 ) )
            else:
                f.write( '{:26.18e} {:26.18e} \n'.format( wfDn[j-canonic.L,i]+noise*random.random(),0.0 ) )
    f.close()

import pickle
def save_pickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10):
    import time
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    max_error : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`np.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao*nao)
    nchol_max = cmax * nao
    # This shape is more convenient for pauxy.
    chol_vecs = np.zeros((nchol_max, nao*nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0,mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2*l+1)*nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0,mol.nbas):
        shls = (i,i+1,0,mol.nbas,i,i+1,0,mol.nbas)
        buf = mol.intor('int2e_sph', shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag:ndiag+di*nao] = buf.reshape(di*nao,di*nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs."%nchol_max)
        print("# max number of cholesky vectors = %d"%nchol_max)
        print("# iteration %5d: delta_max = %f"%(0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao*nao)
    # ERI[:,jl]
    eri_col = mol.intor('int2e_sph',
                         shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
    cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
    chol_vecs[0] = np.copy(eri_col[:,:,cj,cl].reshape(nao*nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor('int2e_sph',
                            shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
        # Select correct ERI chunk from shell.
        cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
        Munu0 = eri_col[:,:,cj,cl].reshape(nao*nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[:nchol+1,nu], chol_vecs[:nchol+1,:])
        chol_vecs[nchol+1] = (Munu0 - R) / (delta_max)**0.5
        #
        nchol += 1
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print ("# iteration %5d: delta_max = %13.8e: time = %13.8e"%info)

    return chol_vecs[:nchol]
    
class CanonicalBais:

    def __init__(self, mol, rhf, lindep=1e-8):

        self.nbasis = mol.nao_nr()
        ovlp   = rhf.get_ovlp()

        value, vector = np.linalg.eigh( ovlp )
        print( "Eigenvalue of overlap matrix: " )
        for index, item in enumerate( value ):
            print( "{:<9} {:26.18e}".format(index, item) )

        if( lindep >= value[-1] ):
            print("Error!!! lindep = {:.12f}, too big for determining the dependency!".format(lindep))
            sys.exit(1)
        numberOfDependent = next(i for i, v in enumerate(value) if v > lindep)

        print( "Number of dependent obritals is {}.".format(numberOfDependent) )
        print('\n')

        self.L = self.nbasis - numberOfDependent

        value = value[numberOfDependent:self.nbasis]
        vector = vector[:,numberOfDependent:self.nbasis]
        sqrtValue = np.sqrt(value)
        self.X = vector / sqrtValue
        self.XInv = vector.T * sqrtValue[:, None]
        self.XT = self.X.T
        #
        # print("HHHHHHHHHHHHHHHH: ", self.X.conj().T @ self.X)
        # X = rhf.mo_coeff
        # print("sssssssssssssss: ", X.conj().T @ X)
        # X = rhf.mo_coeff
        # self.X = X
        # self.XInv = aux_inv=np.linalg.inv(X)
        # self.XT = self.X.T

def getCholeskyAO(mol=None, tol=1e-8):

    nbasis  = mol.nao_nr()
    eri = scf._vhf.int2e_sph(mol._atm,mol._bas,mol._env)
    V   = ao2mo.restore(1, eri, nbasis)
    V   = V.reshape( nbasis*nbasis, nbasis*nbasis )

    choleskyVecAO = []; choleskyNum = 0
    Vdiag = V.diagonal().copy()
    while True:
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        print( "Inside modified Cholesky {:<9} {:26.18e}.".format(choleskyNum, vmax) )
        if(vmax<tol or choleskyNum==nbasis*nbasis):
            print( "Number of Cholesky fields is {:9}".format(choleskyNum) )
            print('\n')
            break
        else:
            oneVec = V[imax]/np.sqrt(vmax)
            #
            choleskyVecAO.append( oneVec )
            choleskyNum+=1
            V -= np.dot(oneVec[:, None], oneVec[None,:])
            Vdiag -= oneVec**2
        #
    return choleskyNum, choleskyVecAO

def getCholeskyMO_AO(mol=None, canonic=None, tol=1e-8):

    nbasis  = mol.nao_nr()
    choleskyNum, choleskyVecAO = getCholeskyAO(mol, tol)

    choleskyVecMO = np.zeros((choleskyNum, canonic.L*canonic.L))
    for i in range(choleskyNum):
        oneVec = choleskyVecAO[i].reshape(nbasis, nbasis)
        choleskyVecMO[i] = np.dot( canonic.XT, np.dot( oneVec, canonic.X ) ).ravel()

    return choleskyNum, choleskyVecAO, choleskyVecMO

##################################################################
def setupMolecule_icf(atoms=None,chrg=None,spn=None,basis=None,psp=None,sym=None):

    mol          = pyscf.gto.Mole()
    mol.verbose  = 4
    mol.output   = 'mole.dat'
    mol.atom     = atoms
    mol.charge   = chrg
    mol.spin     = spn
    mol.basis    = basis
    mol.symmetry = sym
    mol.ecp      = psp
    mol.unit     = 'Angstrom'
    mol.build()

    Enuc    = gto.energy_nuc(mol)
    nbasis  = mol.nao_nr()
    nelec_a = (mol.nelectron+mol.spin)//2
    nelec_b = (mol.nelectron-mol.spin)//2

    print('Molecule [geometry in Angstrom]')
    print(atoms)

    print('Nuclear repulsion energy = {:26.18e} '.format(Enuc))

    print('AO basis ',basis)
    basis_label = gto.spheric_labels(mol)
    for index, item in enumerate( basis_label ):
        print( "{:<9} {:<16}".format(index, item) )

    print('charge          {:>9d}'.format(chrg)   )
    print('spin            {:>9d}'.format(spn)    )
    print('orbitals        {:>9d}'.format(nbasis) )
    print('alpha electrons {:>9d}'.format(nelec_a))
    print('beta  electrons {:>9d}'.format(nelec_b))
    print('\n')

    return mol


def writeInputForModel_forGeneralHamiltonian(scale=1.0, mol=None, rhf=None, canonic=None, tol=1e-8, name="model_param"):

    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2

    choleskyNum, choleskyVecMO = getCholeskyMO(mol, canonic, tol)
    #
    choleskyVecMO = choleskyVecMO * np.sqrt(scale)
    t = np.dot( canonic.XT, np.dot( rhf.get_hcore(), canonic.X ) ) * scale
    #
    K = t.copy()
    for i in range(choleskyNum):
        oneVec = choleskyVecMO[i].reshape(canonic.L, canonic.L)
        K += (-0.5)*np.dot( oneVec, oneVec )
    ###############################
    svdVecs=choleskyVecMO
    svdNumber=choleskyNum
    ############################
    #ATTention: there is a transpose between python and c++ i/o trans
    KT=np.zeros((2*canonic.L, 2*canonic.L),dtype=np.complex128)
    KT[0:canonic.L,0:canonic.L]=K.transpose()
    KT[canonic.L:2*canonic.L,canonic.L:2*canonic.L]=K.transpose()

    svdVecsT=np.zeros((svdNumber, 2*canonic.L, 2*canonic.L),dtype=np.complex128)
    for i in range(svdNumber):
        svdVecsT[i,0:canonic.L,0:canonic.L]=svdVecs[i].reshape(canonic.L, canonic.L).transpose()
        svdVecsT[i,canonic.L:2*canonic.L,canonic.L:2*canonic.L]=svdVecs[i].reshape(canonic.L, canonic.L).transpose()

    f = h5py.File(name, "w")
    f.create_dataset("L",              (1,),                                data=[2*canonic.L],           dtype='int')
    f.create_dataset("Nup",              (1,),                                data=[Nup],           dtype='int')
    f.create_dataset("Ndn",              (1,),                                data=[Ndn],           dtype='int')
    f.create_dataset("N",            (1,),                                data=[Nup+Ndn],                     dtype='int')
    f.create_dataset("svdNumber", (1,),                                data=[svdNumber],           dtype='int')
    f.create_dataset("K_r",              ((2*canonic.L)**2,),                 data=KT.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_r",   (svdNumber*(2*canonic.L)**2,), data=svdVecsT.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdBg_r",     (svdNumber,), data=np.zeros(svdNumber),   dtype='float64')
    f.create_dataset("K_i",              ((2*canonic.L)**2,),                 data=KT.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_i",   (svdNumber*(2*canonic.L)**2,), data=svdVecsT.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdBg_i",     (svdNumber,),                  data=np.zeros(svdNumber),   dtype='float64')
    f.close()

def writeInputForModel_forGeneralHamiltonian_HAFQMC(mol=None, rhf=None, canonic=None, tol=1e-8, hamiltonian_path="AFQMC_hamiltonian.pkl", name="model_param"):
    Nup = (mol.nelectron+mol.spin)//2
    Ndn = (mol.nelectron-mol.spin)//2
    choleskyNum, choleskyVecAO, choleskyVecMO = getCholeskyMO_AO(mol, canonic, tol)
    #
    t = np.dot( canonic.XT, np.dot( rhf.get_hcore(), canonic.X ) )
    #
    K = t.copy()
    for i in range(choleskyNum):
        oneVec = choleskyVecMO[i].reshape(canonic.L, canonic.L)
        K += (-0.5)*np.dot( oneVec, oneVec )
    ###############################
    wfn_a_icf = rhf.mo_coeff[:, 0:Nup]
    wfn_b_icf = rhf.mo_coeff[:, 0:Ndn]
    wfn_a_icf = np.dot( canonic.XInv, wfn_a_icf )
    wfn_b_icf = np.dot( canonic.XInv, wfn_b_icf )
    #
    choleskyVecMO_matrix=np.array(choleskyVecMO)
    choleskyVecMO_matrix = choleskyVecMO_matrix.reshape(choleskyNum, mol.nao, mol.nao)
    ###############################
    svdVecs=choleskyVecMO
    svdNumber=choleskyNum
    ############################
    #ATTention: there is a transpose between python and c++ i/o trans
    KT=np.zeros((2*canonic.L, 2*canonic.L),dtype=np.complex128)
    KT[0:canonic.L,0:canonic.L]=K.transpose()
    KT[canonic.L:2*canonic.L,canonic.L:2*canonic.L]=K.transpose()
    #
    svdVecsT=np.zeros((svdNumber, 2*canonic.L, 2*canonic.L),dtype=np.complex128)
    for i in range(svdNumber):
        svdVecsT[i,0:canonic.L,0:canonic.L]=svdVecs[i].reshape(canonic.L, canonic.L).transpose()
        svdVecsT[i,canonic.L:2*canonic.L,canonic.L:2*canonic.L]=svdVecs[i].reshape(canonic.L, canonic.L).transpose()
    #
    f = h5py.File(name, "w")
    f.create_dataset("L",              (1,),                                data=[2*canonic.L],           dtype='int')
    f.create_dataset("Nup",              (1,),                                data=[Nup],           dtype='int')
    f.create_dataset("Ndn",              (1,),                                data=[Ndn],           dtype='int')
    f.create_dataset("N",            (1,),                                data=[Nup+Ndn],                     dtype='int')
    f.create_dataset("svdNumber", (1,),                                data=[svdNumber],           dtype='int')
    f.create_dataset("K_r",              ((2*canonic.L)**2,),                 data=KT.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_r",   (svdNumber*(2*canonic.L)**2,), data=svdVecsT.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdBg_r",     (svdNumber,), data=np.zeros(svdNumber),   dtype='float64')
    f.create_dataset("K_i",              ((2*canonic.L)**2,),                 data=KT.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_i",   (svdNumber*(2*canonic.L)**2,), data=svdVecsT.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdBg_i",     (svdNumber,),                  data=np.zeros(svdNumber),   dtype='float64')
    f.close()
    #############################
    enuc_icf    = gto.energy_nuc(mol)
    hamil = t, choleskyVecMO_matrix, enuc_icf, (wfn_a_icf, wfn_b_icf), {"orth_mat": canonic.XT.conj().T} 
    save_pickle(hamiltonian_path, hamil)

from pyscf import lib
from pyscf.gto import mole
def basis_change(mol1, mol2, params):
    #convert from the basis of mol1 to mol2, params is the params for mol1
    s22 = mol2.intor_symmetric('int1e_ovlp')
    s21 = mole.intor_cross('int1e_ovlp', mol2, mol1)
    A = lib.cho_solve(s22, s21, strict_sym_pos=False)
    dt = params['params']['ansatz']['wfn_a'].dtype 
    A = A.astype(dt) 
    wfn_a_new = A @ params['params']['ansatz']['wfn_a']
    wfn_b_new = A @ params['params']['ansatz']['wfn_b']
    n2, n1 = A.shape
    A_spin = jnp.block([
    [   A,                jnp.zeros((n2, n1), dtype=dt)],
    [jnp.zeros((n2, n1), dtype=dt),                A   ]])
    hmf_0_new = A_spin @ params['params']['ansatz']['propagators_0']['hmf_ops_0']['hmf'] @ A_spin.conj().T
    hmf_1_new = A_spin @ params['params']['ansatz']['propagators_0']['hmf_ops_1']['hmf'] @ A_spin.conj().T
    vhs_new = jnp.stack([ A_spin @ L1 @ A_spin.conj().T
            for L1 in params['params']['ansatz']['propagators_0']['vhs_ops_0']['vhs'] ], axis=0)
    vhs_new = jnp.asarray(vhs_new, dt)
    params['params']['ansatz']['wfn_a'] = wfn_a_new
    params['params']['ansatz']['wfn_b'] = wfn_b_new
    params['params']['ansatz']['propagators_0']['hmf_ops_0']['hmf'] = hmf_0_new
    params['params']['ansatz']['propagators_0']['hmf_ops_1']['hmf'] = hmf_1_new
    params['params']['ansatz']['propagators_0']['vhs_ops_0']['vhs'] = vhs_new
    return params
def getCholesky_MO(eri,norb, tol=1e-8):

    nbasis  = norb
    V   = ao2mo.restore(1, eri, nbasis)
    V   = V.reshape( nbasis*nbasis, nbasis*nbasis )

    choleskyVecMO = []; choleskyNum = 0
    Vdiag = V.diagonal().copy()
    while True:
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        print( "Inside modified Cholesky {:<9} {:26.18e}.".format(choleskyNum, vmax) )
        if(vmax<tol or choleskyNum==nbasis*nbasis):
            print( "Number of Cholesky fields is {:9}".format(choleskyNum) )
            print('\n')
            break
        else:
            oneVec = V[imax]/np.sqrt(vmax)
            #
            choleskyVecMO.append( oneVec )
            choleskyNum+=1
            V -= np.dot(oneVec[:, None], oneVec[None,:])
            Vdiag -= oneVec**2
        #
    return choleskyNum, choleskyVecMO

def writeInputForModel_for_MO_Hamiltonian_2s_HAFQMC(h1 , h2,  norb, nelec,e_nuc, tol=1e-8, hamiltonian_path="AFQMC_hamiltonian.pkl", name="model_param"):
    Nup, Ndn = nelec  # assumed to be Restricted Hartree-Fock
    choleskyNum,  choleskyVecMO = getCholesky_MO(h2, norb, tol)
    #
    # t = np.dot( canonic.XT, np.dot( rhf.get_hcore(), canonic.X ) )
    t = h1
    #
    K = t.copy()
    for i in range(choleskyNum):
        oneVec = choleskyVecMO[i].reshape(norb, norb)
        K += (-0.5)*np.dot( oneVec, oneVec )

    wfn_a_icf = np.eye(norb)[:, 0:Nup]
    wfn_b_icf = np.eye(norb)[:, 0:Ndn]
    #
    choleskyVecMO_matrix=np.array(choleskyVecMO)
    choleskyVecMO_matrix = choleskyVecMO_matrix.reshape(choleskyNum, norb, norb)
    ###############################
    svdVecs=choleskyVecMO
    svdNumber=choleskyNum
    ############################
    #ATTention: there is a transpose between python and c++ i/o trans
    KT=np.zeros((norb, norb),dtype=np.complex128)
    KT[0: norb,0:norb]=K.transpose()
    # KT[norb:2*norb,norb:2*norb]=K.transpose()
    #
    svdVecsT=np.zeros((svdNumber, norb, norb),dtype=np.complex128)
    for i in range(svdNumber):
        svdVecsT[i,0:norb,0:norb]=svdVecs[i].reshape(norb, norb).transpose()
        # svdVecsT[i,norb:2*norb,norb:2*norb]=svdVecs[i].reshape(norb, norb).transpose()
    #
    f = h5py.File(name, "w")
    f.create_dataset("L",              (1,),                                data=[norb],           dtype='int')
    f.create_dataset("Nup",              (1,),                                data=[Nup],           dtype='int')
    f.create_dataset("Ndn",              (1,),                                data=[Ndn],           dtype='int')
    f.create_dataset("N",            (1,),                                data=[Nup+Ndn],                     dtype='int')
    f.create_dataset("svdNumber", (1,),                                data=[svdNumber],           dtype='int')
    f.create_dataset("K_r",              ((norb)**2,),                 data=KT.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_r",   (svdNumber*(norb)**2,), data=svdVecsT.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdBg_r",     (svdNumber,), data=np.zeros(svdNumber),   dtype='float64')
    f.create_dataset("K_i",              ((norb)**2,),                 data=KT.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_i",   (svdNumber*(norb)**2,), data=svdVecsT.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdBg_i",     (svdNumber,),                  data=np.zeros(svdNumber),   dtype='float64')
    f.close()
    #############################
    enuc_icf    = e_nuc
    hamil = t, choleskyVecMO_matrix, enuc_icf, (wfn_a_icf, wfn_b_icf), {"orth_mat": np.eye(norb)}
    save_pickle(hamiltonian_path, hamil)
