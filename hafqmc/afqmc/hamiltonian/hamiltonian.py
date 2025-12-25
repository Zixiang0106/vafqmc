# In jax_afqmc/hamiltonian.py

import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass
from typing import Tuple, Any

# --------------------------------------------------------------------
# 辅助函数 (完整翻译)
# --------------------------------------------------------------------
@jit
def construct_h1e_mod(chol: jnp.ndarray, h1e: jnp.ndarray) -> jnp.ndarray:
    """构建修正后的单体哈密顿量 (JAX version)。"""
    v0 = 0.5 * jnp.einsum("apr,arq->pq", chol, chol)
    h1e_mod = h1e - v0
    return h1e_mod

@jit
def pack_cholesky(idx1: jnp.ndarray, idx2: jnp.ndarray, chol: jnp.ndarray) -> jnp.ndarray:
    """将Cholesky向量打包成上三角形式 (JAX version)。"""
    # JAX的索引与PyTorch/NumPy相同
    return chol[:, idx1, idx2]

# --------------------------------------------------------------------
# Hamiltonian 数据结构和函数
# --------------------------------------------------------------------

@dataclass
class Hamiltonian:
    """
    一个JAX dataclass，用于持有哈密顿量的所有信息。
    完整翻译自PyTorch版本中的 HamObs 类。
    """
    # 核心物理量
    h1e: jnp.ndarray
    chol: jnp.ndarray
    enuc: float
    
    # 派生或预计算量
    h1e_mod: jnp.ndarray
    packedchol: jnp.ndarray
    idx1: jnp.ndarray # 上三角索引
    idx2: jnp.ndarray # 上三角索引
    
    # 系统和可观测量信息
    nelec0: int
    nao: int
    nchol: int
    obs: Tuple[jnp.ndarray, Any] = None # (矩阵, 常数项)
    obs_type: str = "dipole"
    coupling_shape: Tuple = (1,)


def create_hamiltonian(
    nelec0: int, nao: int, h1e: jnp.ndarray, chol: jnp.ndarray, enuc: float, 
    observable: Tuple = None, obs_type: str = "dipole"
) -> Hamiltonian:
    """
    一个“工厂函数”，用于创建和初始化Hamiltonian对象。
    """
    # 计算修正后的h1e
    h1e_mod = construct_h1e_mod(chol, h1e)
    
    # 获取上三角索引
    idx1, idx2 = jnp.triu_indices(nao)
    
    # 打包Cholesky向量
    packedchol = pack_cholesky(idx1, idx2, chol)
    
    # 获取维度信息
    nchol = chol.shape[0]

    # 设置耦合形状
    coupling_shape = h1e.shape if obs_type == "1rdm" else (1,)

    return Hamiltonian(
        h1e=h1e,
        chol=chol,
        enuc=enuc,
        h1e_mod=h1e_mod,
        packedchol=packedchol,
        idx1=idx1,
        idx2=idx2,
        nelec0=nelec0,
        nao=nao,
        nchol=nchol,
        obs=observable,
        obs_type=obs_type,
        coupling_shape=coupling_shape
    )

@jit
def rot_ham_with_orbs(hamiltonian: Hamiltonian, rot_mat: jnp.ndarray) -> Hamiltonian:
    """用给定的幺正矩阵旋转哈密顿量 (JAX version)"""
    h1e_new = rot_mat.conj().T @ hamiltonian.h1e @ rot_mat
    chol_new = jnp.einsum("qi,aij,jp->aqp", rot_mat.conj(), hamiltonian.chol, rot_mat)
    
    obs_mat_new = rot_mat.conj().T @ hamiltonian.obs[0] @ rot_mat
    obs_new = (obs_mat_new, hamiltonian.obs[1])
    
    # 调用工厂函数创建一个新的、旋转后的Hamiltonian对象
    return create_hamiltonian(
        hamiltonian.nelec0, hamiltonian.nao, h1e_new, chol_new, 
        hamiltonian.enuc, obs_new, obs_type=hamiltonian.obs_type
    )

@jit
def ham_with_obs(hamiltonian: Hamiltonian, coupling: jnp.ndarray) -> Hamiltonian:
    """将可观测量的耦合项加入到单体哈密顿量中 (JAX version)"""
    # 关键区别: JAX函数不应有副作用(in-place modification)。
    # 我们返回一个带有修改后h1e和h1e_mod的新Hamiltonian对象。
    h1e_new = hamiltonian.h1e + coupling * hamiltonian.obs[0]
    h1e_mod_new = hamiltonian.h1e_mod + coupling * hamiltonian.obs[0]
    
    # 使用 replace 功能来高效地创建一个新实例
    from dataclasses import replace
    return replace(hamiltonian, h1e=h1e_new, h1e_mod=h1e_mod_new)