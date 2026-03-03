import jax
from jax import lax
from jax import numpy as jnp
from flax import linen as nn
from typing import Optional, Sequence, Union
from functools import partial

from .utils import _t_real, _t_cplx
from .utils import fix_init, symmetrize, Serial, cmult, scatter
from .utils import warp_spin_expm, make_expm_apply
from .hamiltonian import _align_rdm, calc_rdm


class OneBody(nn.Module):
    init_hmf: jnp.ndarray
    parametrize: bool = False
    init_random: float = 0.
    hermite_out: bool = False
    dtype: Optional[jnp.dtype] = None
    expm_option: Union[str, tuple] = ()

    @property
    def nbasis(self):
        return self.init_hmf.shape[-1]

    def setup(self):
        if self.parametrize:
            self.hmf = self.param("hmf", fix_init, 
                self.init_hmf, self.dtype, self.init_random)
        else:
            self.hmf = self.init_hmf

    def __call__(self, step):
        hmf = symmetrize(self.hmf) if self.hermite_out else self.hmf
        hmf = cmult(step, hmf)
        return hmf
    
    @property
    def expm_apply(self):
        _expm_op = self.expm_option
        _expm_op = (_expm_op,) if isinstance(_expm_op, str) else _expm_op
        return warp_spin_expm(make_expm_apply(*_expm_op))


class AuxField(nn.Module):
    init_vhs: jnp.ndarray
    trial_wfn: Optional[jnp.ndarray] = None
    parametrize: bool = False
    init_random: float = 0.
    hermite_out: bool = False
    dtype: Optional[jnp.dtype] = None
    expm_option: Union[str, tuple] = ()

    @property
    def nbasis(self):
        return self.init_vhs.shape[-1]

    @property
    def nfield(self):
        return self.init_vhs.shape[0]

    def setup(self):
        if self.parametrize:
            self.vhs = self.param("vhs", fix_init, 
                self.init_vhs, self.dtype, self.init_random)
        else:
            self.vhs = self.init_vhs
        self.nhs = self.init_vhs.shape[0]
        self.trial_rdm = (calc_rdm(self.trial_wfn, self.trial_wfn) 
            if self.trial_wfn is not None else None)

    def __call__(self, step, fields, curr_wfn=None):
        vhs = symmetrize(self.vhs) if self.hermite_out else self.vhs
        log_weight = - 0.5 * (fields ** 2).sum()
        if (isinstance(self.expm_option, (list, tuple)) and len(self.expm_option) > 0 and self.expm_option[0] != "diag"):
          if self.trial_rdm is not None:
              vhs, vbar0 = meanfield_subtract(vhs, self.trial_rdm)
              fields += step * vbar0
          # this dynamic shift is buggy, keep it here for reference
          if curr_wfn is not None and self.trial_wfn is not None:
              trdm = calc_rdm(self.trial_wfn, curr_wfn)
              _, vbar = meanfield_subtract(vhs, lax.stop_gradient(trdm), 0.1)
              fshift = step * vbar
              log_weight += - fields @ fshift - 0.5 * (fshift ** 2).sum()
              fields += fshift
        vhs_sum = jnp.tensordot(fields, vhs, axes=1)
        vhs_sum = cmult(step, vhs_sum)
        return vhs_sum, log_weight
    
    @property
    def expm_apply(self):
        _expm_op = self.expm_option
        _expm_op = (_expm_op,) if isinstance(_expm_op, str) else _expm_op
        return warp_spin_expm(make_expm_apply(*_expm_op))


class AuxFieldNet(AuxField):
    hidden_sizes: Optional[Sequence[int]] = None
    actv_fun: str = "gelu"
    zero_init: bool = True
    mod_density: bool = False

    def setup(self):
        super().setup()
        nhs = self.nhs
        last_init =nn.zeros if self.zero_init else nn.initializers.lecun_normal()
        outdim = nhs+1 if self.mod_density else nhs
        self.last_dense = nn.Dense(outdim, param_dtype=self.dtype, 
                                   kernel_init=last_init, bias_init=nn.zeros)
        if self.hidden_sizes:
            inner_init = nn.initializers.orthogonal(scale=1., column_axis=-1)
            self.network = Serial(
                [nn.Dense(
                    ls if ls and ls > 0 else nhs, 
                    param_dtype = _t_real,
                    kernel_init = inner_init,
                    bias_init = nn.zeros) 
                 for ls in self.hidden_sizes],
                skip_cxn = True,
                actv_fun = self.actv_fun)
        else:
            self.network = None
        
    def __call__(self, step, fields, curr_wfn=None):
        vhs = symmetrize(self.vhs) if self.hermite_out else self.vhs
        log_weight = - 0.5 * (fields ** 2).sum()
        tmp = fields
        if self.network is not None:
            tmp = self.network(tmp)
        tmp = self.last_dense(tmp)
        nfields = fields[:self.nhs] + tmp[:self.nhs]
        if self.mod_density:
            log_weight -= tmp[-1]
        if self.trial_rdm is not None:
            vhs, vbar0 = meanfield_subtract(vhs, self.trial_rdm)
            nfields += step * vbar0
        # this dynamic shift is buggy
        if curr_wfn is not None and self.trial_wfn is not None:
            trdm = calc_rdm(self.trial_wfn, curr_wfn)
            _, vbar = meanfield_subtract(vhs, lax.stop_gradient(trdm), 0.1)
            fshift = step * vbar
            log_weight += - nfields @ fshift - 0.5 * (fshift ** 2).sum()
            nfields += fshift
        vhs_sum = jnp.tensordot(nfields, vhs, axes=1)
        vhs_sum = cmult(step, vhs_sum)
        return vhs_sum, log_weight


class TransformerEncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    dropout: float = 0.
    dtype: Optional[jnp.dtype] = _t_real

    @nn.compact
    def __call__(self, hidden):
        deterministic = not self.has_rng("dropout")
        h1 = nn.LayerNorm(param_dtype=self.dtype, dtype=self.dtype)(hidden)
        h1 = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            param_dtype=self.dtype,
            dtype=self.dtype)(h1)
        hidden = hidden + h1
        h2 = nn.LayerNorm(param_dtype=self.dtype, dtype=self.dtype)(hidden)
        mlp_width = max(1, int(self.d_model * self.mlp_ratio))
        h2 = nn.Dense(mlp_width, param_dtype=self.dtype, dtype=self.dtype)(h2)
        h2 = nn.gelu(h2)
        if self.dropout > 0:
            h2 = nn.Dropout(rate=self.dropout)(h2, deterministic=deterministic)
        h2 = nn.Dense(self.d_model, param_dtype=self.dtype, dtype=self.dtype)(h2)
        if self.dropout > 0:
            h2 = nn.Dropout(rate=self.dropout)(h2, deterministic=deterministic)
        return hidden + h2


class AuxFieldTransformer(AuxField):
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    eps: float = 0.02
    clip_value: float = 3.0
    dropout: float = 0.0
    tanh_bound: bool = False
    zero_init: bool = True
    mod_density: bool = False
    net_dtype: Union[str, jnp.dtype] = "float64"
    log_stats: bool = False

    def _resolve_net_dtype(self):
        if isinstance(self.net_dtype, str):
            name = self.net_dtype.lower()
            if name in ("float32", "f32", "fp32"):
                return jnp.float32
            if name in ("float64", "f64", "fp64", "double"):
                return jnp.float64
            raise ValueError(f"unsupported net_dtype: {self.net_dtype}")
        return jnp.dtype(self.net_dtype)

    def setup(self):
        super().setup()
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError(
                f"n_heads must divide d_model, got d_model={self.d_model}, "
                f"n_heads={self.n_heads}")
        if self.n_layers < 0:
            raise ValueError(f"n_layers must be non-negative, got {self.n_layers}")
        if self.mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}")
        self._ndtype = self._resolve_net_dtype()
        self.token_embed = nn.Dense(
            self.d_model,
            param_dtype=self._ndtype,
            dtype=self._ndtype,
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.zeros)
        self.channel_embedding = self.param(
            "channel_embedding",
            nn.initializers.normal(stddev=0.01),
            (self.nhs, self.d_model),
            self._ndtype)
        self.blocks = [
            TransformerEncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                dtype=self._ndtype)
            for _ in range(self.n_layers)]
        last_init = nn.zeros if self.zero_init else nn.initializers.lecun_normal()
        self.delta_head = nn.Dense(
            1,
            param_dtype=self._ndtype,
            dtype=self._ndtype,
            kernel_init=last_init,
            bias_init=nn.zeros)
        if self.mod_density:
            self.density_head = nn.Dense(
                1,
                param_dtype=self._ndtype,
                dtype=self._ndtype,
                kernel_init=last_init,
                bias_init=nn.zeros)

    def _bound_delta(self, delta):
        clipv = jnp.asarray(jnp.abs(self.clip_value), dtype=delta.dtype)
        if self.tanh_bound:
            return clipv * jnp.tanh(delta / jnp.maximum(clipv, 1e-12))
        return jnp.clip(delta, -clipv, clipv)

    def __call__(self, step, fields, curr_wfn=None):
        vhs = symmetrize(self.vhs) if self.hermite_out else self.vhs
        log_weight = -0.5 * (fields ** 2).sum()
        x = fields[:self.nhs]
        x_net = x.astype(self._ndtype)
        hidden = self.token_embed(x_net[:, None]) + self.channel_embedding
        for block in self.blocks:
            hidden = block(hidden)
        delta_raw = self.delta_head(hidden).reshape(-1)
        delta = self._bound_delta(delta_raw)
        delta = delta.astype(x.dtype)
        nfields = x + jnp.asarray(self.eps, dtype=x.dtype) * delta
        if self.mod_density:
            log_weight -= self.density_head(hidden.mean(0)).reshape(()).astype(log_weight.dtype)
        if self.trial_rdm is not None:
            vhs, vbar0 = meanfield_subtract(vhs, self.trial_rdm)
            nfields += step * vbar0
        # this dynamic shift is buggy
        if curr_wfn is not None and self.trial_wfn is not None:
            trdm = calc_rdm(self.trial_wfn, curr_wfn)
            _, vbar = meanfield_subtract(vhs, lax.stop_gradient(trdm), 0.1)
            fshift = step * vbar
            log_weight += - nfields @ fshift - 0.5 * (fshift ** 2).sum()
            nfields += fshift
        vhs_sum = jnp.tensordot(nfields, vhs, axes=1)
        vhs_sum = cmult(step, vhs_sum)
        if self.log_stats:
            clipv = jnp.asarray(jnp.abs(self.clip_value), dtype=delta_raw.dtype)
            clip_rate = jnp.mean((jnp.abs(delta_raw) >= clipv).astype(self._ndtype))
            self.sow("intermediates", "delta_mean", jnp.mean(delta_raw))
            self.sow("intermediates", "delta_std", jnp.std(delta_raw))
            self.sow("intermediates", "delta_maxabs", jnp.max(jnp.abs(delta_raw)))
            self.sow("intermediates", "clip_rate", clip_rate)
        return vhs_sum, log_weight


def meanfield_subtract(vhs, rdm, cutoff=None):
    if rdm.ndim == 3:
        rdm = rdm.sum(0)
    nao = vhs.shape[-1]
    if rdm.shape[-1] == nao * 2:
        rdm = rdm[:nao, :nao] + rdm[nao:, nao:]
    nelec = lax.stop_gradient(rdm).trace().real
    vbar = jnp.einsum("kpq,pq->k", vhs, rdm)
    if cutoff is not None:
        cutoff *= vbar.shape[-1]
        vbar = vbar / (jnp.maximum(jnp.linalg.norm(vbar), cutoff) / cutoff)
    vhs = vhs - vbar.reshape(-1,1,1) * jnp.eye(vhs.shape[-1]) / nelec
    return vhs, vbar


# below are classes and functions for pw basis

class OneBodyPW(nn.Module):
    init_hmf: jnp.array
    kmask: Optional[jnp.array] = None
    parametrize: bool = False
    k_symmetric: bool = False
    init_random: float = 0.
    dtype: Optional[jnp.dtype] = None
    expm_option: Union[str, tuple] = ()

    @property
    def nbasis(self):
        return self.init_hmf.shape[-1]

    def setup(self):
        if self.parametrize:
            if self.k_symmetric:
                raw_hmf, self.kinvidx = jnp.unique(self.init_hmf, return_inverse=True)
            else:
                raw_hmf = self.init_hmf
            self.hmf = self.param("hmf", fix_init, 
                raw_hmf, self.dtype, self.init_random)
        else:
            self.hmf = self.init_hmf
    
    def __call__(self, step):
        hmf = self.hmf
        if self.parametrize and self.k_symmetric:
            hmf = hmf[self.kinvidx]
        hmf = cmult(step, hmf)
        return hmf

    @property
    def expm_apply(self):
        matmul_fn = lambda A, B: jnp.einsum('k,ki->ki', A, B)
        _expm_op = self.expm_option
        _expm_op = (_expm_op,) if isinstance(_expm_op, str) else _expm_op
        return make_expm_apply(*_expm_op, matmul_fn=matmul_fn)


class AuxFieldPW(nn.Module):
    init_vhs: jnp.ndarray
    kmask: jnp.ndarray
    qmask: jnp.ndarray
    parametrize: bool = False
    q_symmetric: bool = False
    init_random: float = 0.
    dtype: Optional[jnp.dtype] = None
    expm_option: Union[str, tuple] = ()

    @property
    def nbasis(self):
        return int(self.kmask.sum().item())

    @property
    def nfield(self):
        return self.init_vhs.shape[0] * 2

    def setup(self):
        if self.q_symmetric and self.parametrize:
            raw_vhs, self.vinvidx = jnp.unique(self.init_vhs, return_inverse=True)
        else:
            raw_vhs = self.init_vhs
        vhs = jnp.tile(raw_vhs, (4, 1)) # for A and B; plus and minus Q
        if self.parametrize:
            self.vhs = self.param("vhs", fix_init, 
                vhs, self.dtype, self.init_random)
        else:
            self.vhs = vhs
        self.nq = self.init_vhs.shape[0]
        self.nhs = self.nq * 2
    
    def __call__(self, step, fields, curr_wfn=None):
        fields = fields.reshape(2, self.nq)
        # fields = fields.at[:, self.nq//2].set(0)
        log_weight = - 0.5 * (fields ** 2).sum()
        vhs = self.vhs
        if self.q_symmetric and self.parametrize:
            vhs = vhs[:, self.vinvidx]
        vplus = jnp.array([1, 1j]) @ (fields * vhs[(0,2), :])   # rho(Q) terms
        vminus = jnp.array([1, -1j]) @ (fields * vhs[(1,3), :]) # rho(-Q) terms
        vsum = vplus + jnp.flip(vminus)
        vsum = cmult(step, vsum)
        # remove constant multiplication at Q = 0
        vsum = vsum.at[self.nq//2].set(0)
        return vsum, log_weight
    
    @property
    def expm_apply(self):
        from jax.scipy.signal import convolve
        from .utils import fftconvolve
        # sum over all Q for one electron
        def conv1ele(vhs, wfn):
            dtype = vhs.real.dtype
            vq_mesh = scatter(vhs, self.qmask)
            wk_mesh = scatter(wfn, self.kmask)
            nwk_mesh = convolve(vq_mesh, wk_mesh, 'valid')
            # nwk_mesh = fftconvolve(vq_mesh, wk_mesh, 'valid')
            return nwk_mesh[self.kmask]
        # map it for all electrons (at the last axis)
        matmul_fn = jax.vmap(conv1ele, in_axes=(None, -1), out_axes=-1)
        _expm_op = self.expm_option
        _expm_op = (_expm_op,) if isinstance(_expm_op, str) else _expm_op
        return make_expm_apply(*_expm_op, matmul_fn=matmul_fn)
