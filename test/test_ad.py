# test_ad.py
import jax.numpy as jnp
from pyscf import gto, scf

from ad_afqmc import config, utils, wavefunctions, driver

# 1) 强制 x64（必须在创建 jax 数组前）
config.afqmc_config["single_precision"] = False
config.setup_jax()
MPI = config.setup_comm()

# 2) pyscf mean-field
mol = gto.M(
    atom="H 0 0 0; H 0 0 0.74",
    basis="sto-3g",
    charge=0,
    spin=0,
    verbose=0,
)
mf = scf.RHF(mol).run()

# 3) 生成 cholesky + trial coeff
pyscf_prep = utils.prep_afqmc(mf, chol_cut=1e-6, write_to_disk=False)

nelec_tot, nbasis, ms, nchol = pyscf_prep["header"].astype(int)
nalpha = (nelec_tot + ms) // 2
nbeta  = (nelec_tot - ms) // 2
nelec_sp = (nalpha, nbeta)

h0 = float(pyscf_prep["energy_core"])
h1 = jnp.array(pyscf_prep["hcore"]).reshape(nbasis, nbasis)
chol = jnp.array(pyscf_prep["chol"]).reshape(nchol, nbasis, nbasis)

ham, ham_data = utils.set_ham(nbasis, h0, h1, chol, ene0=0.0)

# 4) RHF trial
trial_coeffs = pyscf_prep["trial_coeffs"]
trial = wavefunctions.rhf(nbasis, nelec_sp, n_chunks=1, projector=None)
wave_data = {"mo_coeff": jnp.array(trial_coeffs[0][:, :nalpha])}

# 5) options（对齐你 vafqmc 的 dt/steps）
options = utils.get_options({
    "dt": 0.01,
    "n_walkers": 50,
    "n_prop_steps": 50,
    "n_blocks": 20,

    "n_ene_blocks": 1,
    "n_sr_blocks": 1,
    "n_qr_blocks": 1,

    "n_ene_blocks_eql": 1,
    "n_sr_blocks_eql": 1,
    "n_eql": 50,

    "walker_type": "restricted",
    "trial": "rhf",
    "symmetry_projector": None,
    "free_projection": False,

    # 明确关闭混合精度
    "vhs_mixed_precision": False,
    "trial_mixed_precision": False,
})

prop = utils.set_prop(options)
sampler = utils.set_sampler(options)

# 6) run
e, err = driver.afqmc_energy(
    ham_data, ham, prop, trial, wave_data, sampler, options, MPI
)
print("E, err:", e, err)
