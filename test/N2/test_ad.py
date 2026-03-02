from pyscf import gto, scf
from ad_afqmc import afqmc

mol =  gto.M(atom ="""
    N        0.0000000000      0.0000000000      0.0000000000
    N        3.6000000000      0.0000000000      0.0000000000
    """,
    spin = 0,
    basis = 'ccpvdz',
    unit = 'B')

# UHF
mf = scf.UHF(mol)
mf.kernel()

# afqmc @ UHF
af = afqmc.AFQMC(mf)
af.n_walkers = 1200 # Number of walkers
af.n_blocks = 100 # Number of blocks
af.kernel()