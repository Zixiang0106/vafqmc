# In your new file, e.g., `jax_afqmc/propagation.py`

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial 

from .walkers import Walkers
from .trial import SDTrial, calc_force_bias, calc_overlap
from .hamiltonian import Hamiltonian 


@partial(jit, static_argnames=("nao",))
def construct_vhs(chol: jnp.ndarray, xshifted: jnp.ndarray, dt: float, nao: int):
    isqrtt = jnp.sqrt(dt, dtype=jnp.complex128)
    vhs = isqrtt * jnp.einsum('wi,ijk->wjk', xshifted, chol)
    
    return vhs

@jit
def compute_mf_shift(chol: jnp.ndarray, G: jnp.ndarray) -> jnp.ndarray:
    mf_shift = 1j * jnp.einsum("pij,ji->p", chol, G) 
    return mf_shift

@jit
def compute_expH1(h1e_mod: jnp.ndarray, chol: jnp.ndarray, mf_shift: jnp.ndarray, dt: float) -> jnp.ndarray:
    h1e_mf = h1e_mod + jnp.einsum("p,pij->ij", mf_shift.imag, chol)
    return jax.scipy.linalg.expm(-0.5 * dt * h1e_mf)



@partial(jit, static_argnames=("taylor_order",))
def propagate_step(
    walkers: Walkers, 
    hamiltonian: Hamiltonian, 
    trial: SDTrial, 
    dt: float,
    energy_shift: float,
    rng_key: jnp.ndarray,
    taylor_order: int = 6
):
    key, subkey = jax.random.split(rng_key)
    
    ovlp_old_matrix = calc_overlap(trial, walkers)
    sign_old, logdet_old = jnp.linalg.slogdet(ovlp_old_matrix)
    
    mf_shift = compute_mf_shift(hamiltonian.chol, trial.G)
    expH1 = compute_expH1(hamiltonian.h1e_mod, hamiltonian.chol, mf_shift, dt)
    
    vbias = calc_force_bias(trial, walkers)
    xbar = -jnp.sqrt(dt) * (1j * vbias - mf_shift)
    # (此处省略了对力偏置的边界限制，可以后续添加)

    # --- 2. 传播过程 ---
    # a. 前半步单体传播
    phi_half_propagated = jnp.einsum("ij,wjk->wik", expH1, walkers.walker_states)
    
    # b. 二体传播
    x_sample = jax.random.normal(subkey, shape=xbar.shape, dtype=jnp.float64)
    xshifted = x_sample - xbar.real # 注意 PyTorch 代码中的 xbar 包含了复数，这里根据公式应为实数部分
    
    vhs = construct_vhs(hamiltonian.chol, xshifted, dt, hamiltonian.h1e.shape[0])

    # 使用泰勒展开近似矩阵指数 e^V, V = vhs
    # 注意：这是一个批处理操作，vhs的shape是(n_walkers, n_basis, n_basis)
    phi_v_propagated = phi_half_propagated
    Temp = phi_half_propagated
    for n in range(1, taylor_order + 1):
        Temp = jnp.einsum("wij,wjk->wik", vhs, Temp) / n
        phi_v_propagated = phi_v_propagated + Temp
        
    # c. 后半步单体传播
    phi_new = jnp.einsum("ij,wjk->wik", expH1, phi_v_propagated)

    # --- 3. 权重更新 ---
    ovlp_new_matrix = calc_overlap(trial, Walkers(walker_states=phi_new, walker_weights=walkers.walker_weights))
    sign_new, logdet_new = jnp.linalg.slogdet(ovlp_new_matrix)
    
    # 对数重叠比率
    log_overlap_ratio = (logdet_new + jnp.log(sign_new)) - (logdet_old + jnp.log(sign_old))
    
    # 重要性采样项
    log_importance_term = jnp.sum(x_sample * xbar.real - 0.5 * xbar.real * xbar.real, axis=-1)
    
    # 平均场项
    log_mf_term = -jnp.sqrt(dt) * jnp.einsum("wi,i->w", xshifted, mf_shift)
    
    # 常数项 (能量偏移)
    h0shift = hamiltonian.enuc - 0.5 * jnp.dot(mf_shift.imag, mf_shift.imag)
    log_const_term = dt * (energy_shift - h0shift)

    # 组合所有项
    log_weight_update = (
        log_overlap_ratio.real # 取实部
        + log_importance_term 
        + log_mf_term.real # 取实部
        + log_const_term.real # 取实部
    )
    
    # 相位约束 (Phaseless aAFQMC)
    dtheta = (log_overlap_ratio + log_mf_term).imag
    phaseless_factor = jnp.maximum(0.0, jnp.cos(dtheta))

    weights_new = walkers.walker_weights * jnp.exp(log_weight_update) * phaseless_factor

    # --- 4. 返回新状态 ---
    new_walkers = Walkers(walker_states=phi_new, walker_weights=weights_new)
    
    return new_walkers, key