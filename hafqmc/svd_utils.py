import os
import sys; sys.path.append( os.environ['AFQMCLAB_DIR']+"/scripts/pyscf" )
from molecule import *
from rhf import *
from uhf import *
from model import *
import numpy as np

import pyscf
from   pyscf  import scf

###################################################################################
#math related functions
###################################################################################
def matrix_exponential(matrix, n_terms):
    n_terms = n_terms +1
    n = matrix.shape[0]  
    result = np.eye(n, dtype=np.complex128) 
    current_term = np.eye(n, dtype=np.complex128)  
    
    for k in range(1, n_terms):
        current_term = np.dot(current_term, matrix) / k
        result += current_term
    
    return result
###################################################################################
#PKL related functions
###################################################################################
import pickle
def save_pickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
        
def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
        

###################################################################################
#Fcidump related functions
###################################################################################
def fcidump_header(nel, norb, spin):
    header = (
        "&FCI " +
        "NORB={:d}, ".format(norb) +
        "NELEC={:d}, ".format(nel) +
        "MS2={:d},\n".format(spin) +
        "ORBSYM=" +
        ",".join([str(1)]*norb) +
        ",\n" +
        "ISYM=1\n" +
        "&END\n"
        )
    return header

def read_fcidump_header(filename, mline=1024):
  meta = dict()
  found_end = False
  with open(filename) as f:
    for iline in range(mline):
      line = f.readline()
      if 'END' in line or '/' in line:
        found_end = True
        break
      for i in line.split(','):
        if 'NORB' in i:
          nbasis = int(i.split('=')[1])
          meta['nbasis'] = nbasis
        elif 'NELEC' in i:
          nelec = int(i.split('=')[1])
          meta['nelec'] = nelec
        elif 'MS2' in i:
          ms2 = int(i.split('=')[1])
          meta['ms2'] = ms2
  if not found_end:
    msg = "FCIDUMP header longer than %d lines" % mline
    raise RuntimeError(msg)
  return meta

def read_fcidump(filename, symmetry=8, verbose=True):
    """Read in integrals from file.

    Parameters
    ----------
    filename : string
        File containing integrals in FCIDUMP format.
    symmetry : int
        Permutational symmetry of two electron integrals.
    verbose : bool
        Controls printing verbosity. Optional. Default: False.

    Returns
    -------
    h1e : :class:`numpy.ndarray`
        One-body part of the Hamiltonian.
    h2e : :class:`numpy.ndarray`
        Two-electron integrals.
    ecore : float
        Core contribution to the total energy.
    nelec : tuple
        Number of electrons.
    """
    assert(symmetry==1 or symmetry==4 or symmetry==8)
    meta = read_fcidump_header(filename)
    nbasis = meta['nbasis']
    nelec = meta['nelec']
    ms2 = meta['ms2']
    if verbose:
        print ("# Reading integrals in plain text FCIDUMP format.")
    with open(filename) as f:
        while True:
            line = f.readline()
            if 'END' in line or '/' in line:
                break
        if verbose:
            print("# Number of orbitals: {}".format(nbasis))
            print("# Number of electrons: {}".format(nelec))
        h1e = np.zeros((nbasis, nbasis), dtype=np.complex128)
        h2e = np.zeros((nbasis, nbasis, nbasis, nbasis), dtype=np.complex128)
        lines = f.readlines()
        for l in lines:
            s = l.split()
            # ascii fcidump uses Chemist's notation for integrals.
            # each line contains v_{ijkl} i k j l
            # Note (ik|jl) = <ij|kl>.
            if l.strip().startswith('('):  # parenthesis complex value
                left, right = l.split(')')
                s = right.split()
                rt, it = left.split(',')
                int_r = float(rt.replace('(', ''))
                int_i = float(it)
                integral = int_r+1j*int_i
            elif len(s) == 6:
                # FCIDUMP from quantum package.
                integral = float(s[0]) + 1j*float(s[1])
                s = s[1:]
            else:
                try:
                    integral = float(s[0])
                except ValueError:
                    ig = ast.literal_eval(s[0].strip())
                    integral = ig[0] + 1j*ig[1]
            i, k, j, l = [int(x) for x in s[-4:]]
            if i == j == k == l == 0:
                ecore = integral
            elif j == 0 and l == 0:
                # <i|k> = <k|i>
                h1e[i-1,k-1] = integral
                h1e[k-1,i-1] = integral.conjugate()
            elif i > 0  and j > 0 and k > 0 and l > 0:
                # Assuming 8 fold symmetry in integrals.
                # <ij|kl> = <ji|lk> = <kl|ij> = <lk|ji> =
                # <kj|il> = <li|jk> = <il|kj> = <jk|li>
                # (ik|jl)
                h2e[i-1,k-1,j-1,l-1] = integral
                if symmetry == 1:
                    continue
                # (jl|ik)
                h2e[j-1,l-1,i-1,k-1] = integral
                # (ki|lj)
                h2e[k-1,i-1,l-1,j-1] = integral.conjugate()
                # (lj|ki)
                h2e[l-1,j-1,k-1,i-1] = integral.conjugate()
                if symmetry == 4:
                    continue
                # (ki|jl)
                h2e[k-1,i-1,j-1,l-1] = integral
                # (lj|ik)
                h2e[l-1,j-1,i-1,k-1] = integral
                # (ik|lj)
                h2e[i-1,k-1,l-1,j-1] = integral
                # (jl|ki)
                h2e[j-1,l-1,k-1,i-1] = integral
    if symmetry == 8:
        if np.any(np.abs(h1e.imag)) > 1e-18:
            print("# Found complex numbers in one-body Hamiltonian but 8-fold"
                  " symmetry specified.")
        if np.any(np.abs(h2e.imag)) > 1e-18:
            print("# Found complex numbers in two-body Hamiltonian but 8-fold"
                  " symmetry specified.")
    nalpha = (nelec + ms2) // 2
    nbeta = nalpha - ms2
    return h1e, h2e, ecore, (nalpha, nbeta) 
##################################################################
#Pyscf related functions 
##################################################################
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


def getCholeskyMO(eri,norb, tol=1e-8):

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

##################################################################
#Low rank related functions 
##################################################################
def write_rhf_density_matrix(elec,rhf):
    Nup, Ndn = elec
    # Calculate and save the RHF density matrix (1rdm)
    if Nup != Ndn:
        dm_rhf = rhf.make_rdm1(ao_repr=True)
        np.save("rhf_density_matrix.npy", dm_rhf)
        print("================================================")
        print("RHF density matrix saved to rhf_density_matrix.npy: ",dm_rhf.shape)
        print("================================================")
    else:
        dm_rhf_up = rhf.make_rdm1(ao_repr=True)
        dm_rhf = np.stack([dm_rhf_up, dm_rhf_up])
        np.save("rhf_density_matrix.npy", dm_rhf)
        print("================================================")
        print("RHF density matrix saved to rhf_density_matrix.npy: ",dm_rhf.shape)
        print("================================================")
#############
def write_mp2_density_matrix(elec,rhf):
    Nup, Ndn = elec
    from pyscf import mp
    # 1. 初始化 MP2 对象（ROHF 自动适配为 ROMP2）
    mp2 = mp.MP2(rhf)  # 等价于 pyscf.mp.mp2.ROMP2(rohf)

    # （可选）设置 MP2 参数（冻结核心、收敛阈值等）
    mp2.frozen = 0  # 冻结 1s 核心轨道（减少计算量，精度损失可忽略）
    mp2.conv_tol = 1e-10  # 关联能收敛阈值

    # 2. 执行 MP2 计算（用 run() 方法，自动存储结果到 mp2 对象）
    mp2.kernel() # 无需手动接收返回值，结果已存入 mp2.e_corr（关联能）、mp2.t2（双激发振幅）

    # 打印 MP2 关键结果
    print("=== MP2 计算结果 ===")
    print(f"ROHF 能量：{rhf.e_tot:.6f} Ha")
    print(f"MP2 关联能：{mp2.e_corr:.6f} Ha")
    print(f"MP2 总能量：{rhf.e_tot + mp2.e_corr:.6f} Ha\n")

    # 3. 提取 MP2 自旋分辨 1-RDM（自动读取 mp2.t2）
    if Nup != Ndn:
        rdm1_mp2 = mp2.make_rdm1(ao_repr=True)  # 无需传参，直接调用
        np.save("romp2_density_matrix.npy", rdm1_mp2)
        print("================================================")
        print("romp2 density matrix saved to romp2_density_matrix.npy: ",rdm1_mp2[0].shape,rdm1_mp2[1].shape)
        print("================================================")
    else:
        rdm1_mp2_up = mp2.make_rdm1(ao_repr=True)  # 无需传参，直接调用
        rdm1_mp2 = np.stack([rdm1_mp2_up, rdm1_mp2_up])
        np.save("romp2_density_matrix.npy", rdm1_mp2)
        print("================================================")
        print("romp2 density matrix saved to romp2_density_matrix.npy: ",rdm1_mp2.shape)
        print("================================================")
#############

def write_PKL_with_full_format_from_pyscf(mol=None, rhf=None, canonic=None, tol=1e-8, hamiltonian_path="AFQMC_hamiltonian.pkl"):
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
    choleskyVecMO_matrix = choleskyVecMO.reshape(choleskyNum, mol.nao, mol.nao)
    #############################
    enuc_icf    = gto.energy_nuc(mol)
    hamil = t, choleskyVecMO_matrix, enuc_icf, (wfn_a_icf, wfn_b_icf), {"orth_mat": canonic.XT.conj().T} 
    save_pickle(hamiltonian_path, hamil)

def write_PKL_with_lowRank_format_from_pyscf(mol=None, rhf=None, canonic=None, max_nhs=100, tol=1e-8, hamiltonian_path="AFQMC_hamiltonian.pkl"):
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
    choleskyVecMO_matrix = choleskyVecMO.reshape(choleskyNum, mol.nao, mol.nao)
    ###############################
    # 对每个cholesky向量进行对角化
    ###############################
    print("============================================================")
    print("SVD for choleskyVecMO_matrix to get low rank representation for HAFQMC optimization:")
    print("============================================================")
    # 
    if max_nhs > choleskyNum:
        max_nhs = choleskyNum
    choleskyVecMO_matrix_U0 = np.eye(mol.nao)
    choleskyVecMO_matrix_D = choleskyVecMO_matrix[:max_nhs]
    choleskyVecMO_matrix_V0dagger = np.eye(mol.nao)
    ##
    t_U = np.eye(mol.nao)
    t_D = t
    t_Vdagger = np.eye(mol.nao)
    #############################
    enuc_icf    = gto.energy_nuc(mol)
    hamil = t, choleskyVecMO_matrix, t_U, t_D, t_Vdagger, choleskyVecMO_matrix_U0, choleskyVecMO_matrix_D, choleskyVecMO_matrix_V0dagger, enuc_icf, (wfn_a_icf, wfn_b_icf), {"orth_mat": canonic.XT.conj().T} 
    save_pickle(hamiltonian_path, hamil)

def setupMolecule_icf(atoms=None,chrg=None,spn=None,basis=None,psp=None,sym=None,unit='Angstrom'):

    mol          = pyscf.gto.Mole()
    mol.verbose  = 4
    mol.output   = 'mole.dat'
    mol.atom     = atoms
    mol.charge   = chrg
    mol.spin     = spn
    mol.basis    = basis
    mol.symmetry = sym
    mol.ecp      = psp
    mol.unit     = unit
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
#################################################


# AFQMC input
def write_modelParam_lowRank_SD1s_from_PKL_lowRank_SD1s(hamiltonian_path="AFQMC_hamiltonian.pkl", name="model_param"):
    hamil = load_pickle(hamiltonian_path)

    # see https://github.com/y1xiaoc/hafqmc/blob/master/hafqmc/hamiltonian.py#L334
    # for detailed meaning of the arrays
    t, choleskyVecMO, h1e_U, h1e_D, h1e_Vdagger, ceri_U0, ceri_D, ceri_V0dagger, enuc, (wfn_a, wfn_b), aux = hamil
    # 
    L = t.shape[0]
    Nup = wfn_a.shape[1]
    Ndn = wfn_b.shape[1]
    # 
    choleskyNum = choleskyVecMO.shape[0]
    #
    K = t.copy()
    for i in range(choleskyNum):
        oneVec = choleskyVecMO[i,:,:]
        K += (-0.5)*np.dot( oneVec, oneVec )
    ###############################
    ###############################
    svdVecs=choleskyVecMO
    svdNumber=choleskyNum
    ############################
    #ATTention: there is a transpose between python and c++ i/o trans
    KT=np.zeros((L, L),dtype=np.complex128)
    KT[0:L,0:L]=K.transpose()
    #
    svdVecsT=np.zeros((svdNumber, L, L),dtype=np.complex128)
    for i in range(svdNumber):
        svdVecsT[i,0:L,0:L]=svdVecs[i].reshape(L, L).transpose()
    #
    f = h5py.File(name, "w")
    f.create_dataset("L",              (1,),                                data=[L],           dtype='int')
    f.create_dataset("SD2sL",              (1,),                                data=[L],           dtype='int')
    f.create_dataset("truncatedDup",              (1,),                       data=[0],           dtype='int')
    f.create_dataset("truncatedDdn",              (1,),                       data=[0],           dtype='int')
    f.create_dataset("Nup",              (1,),                                data=[Nup],           dtype='int')
    f.create_dataset("Ndn",              (1,),                                data=[Ndn],           dtype='int')
    f.create_dataset("N",            (1,),                                data=[Nup+Ndn],                     dtype='int')
    f.create_dataset("svdNumber", (1,),                                data=[svdNumber],           dtype='int')
    f.create_dataset("truncatedSvdNumber", (1,),                                data=[0],           dtype='int')
    f.create_dataset("K_r",              ((L)**2,),                 data=KT.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_r",   (svdNumber*(L)**2,), data=svdVecsT.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                                   
    f.create_dataset("svdBg_r",     (svdNumber,), data=np.zeros(svdNumber),   dtype='float64')
    f.create_dataset("K_i",              ((L)**2,),                 data=KT.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_i",   (svdNumber*(L)**2,), data=svdVecsT.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdBg_i",     (svdNumber,),                  data=np.zeros(svdNumber),   dtype='float64')
    f.close()
    #############################


##################################################################
# Hamiltonian for model_Jastrow_param
##################################################################
# VAFQMC SD1s to AFQMC SD1s
##################################################################
def write_modelJastrowParam_lowRank_SD1s_from_checkpoint_lowRank_SD1s_slice1(hamiltonian_path="hamiltonian.pkl", param_path="checkpoint.pkl",  name="model_Jastrow_param"):
    #|\phiT> = \prod [exp(-tT)exp(-tV)]|\phi_0>
        
    # also 2.4, 2.7, 3.0, 3.6, 4.2
    hamil = load_pickle(hamiltonian_path)
    ckpt = load_pickle(param_path)
    params = ckpt[1][1]['params']['ansatz']

    # see https://github.com/y1xiaoc/hafqmc/blob/master/hafqmc/hamiltonian.py#L334
    # for detailed meaning of the arrays
    h1e, ceri, h1e_U, h1e_D, h1e_Vdagger, ceri_U, ceri_D, ceri_Vdagger, enuc, (wfn_a, wfn_b), aux = hamil
    # 
    L = h1e.shape[0]
    Nup = wfn_a.shape[1]
    Ndn = wfn_b.shape[1]
    # basis size 28
    # number of electrons 14 (7 up + 7 down)
    print('shape of the hamiltonian arrays:')
    print('one body operator K (h1e):', h1e.shape)
    print('two body operator after cholesky (ceri):', ceri.shape)
    print('nuclear contribution of energy (enuc):', enuc)
    print('spin up wfn coeffs (wfn_a):', wfn_a.shape)
    print('spin down wfn coeffs (wfn_b):', wfn_b.shape)
    print('ao2mo rotaton matrix ("orth_mat"):', aux['orth_mat'].shape)

    # remove extra layers
    # params = params['params']['ansatz']

    # one body operators at different slices, in GHF
    hmf=[]
    hmf0_U = params['propagators_0']['hmf_ops_0']['hmf_U']
    hmf0_D = params['propagators_0']['hmf_ops_0']['hmf_D']
    hmf0_Vdagger = params['propagators_0']['hmf_ops_0']['hmf_Vdagger']
    hmf0 = hmf0_U @ hmf0_D @ hmf0_Vdagger
    hmf.append(hmf0)
    hmf1_U = params['propagators_0']['hmf_ops_1']['hmf_U']
    hmf1_D = params['propagators_0']['hmf_ops_1']['hmf_D']
    hmf1_Vdagger = params['propagators_0']['hmf_ops_1']['hmf_Vdagger']
    hmf1 = hmf1_U @ hmf1_D @ hmf1_Vdagger
    hmf.append(hmf1)
    print(f"{hmf0.shape = }")

    # decompsed two body operators, same for all slices, 100 aux fields
    vhs=[]
    vhs1_U0 = params['propagators_0']['vhs_ops_0']['vhs_U0']
    vhs1_D = params['propagators_0']['vhs_ops_0']['vhs_D']
    vhs1_Vdagger0 = params['propagators_0']['vhs_ops_0']['vhs_V0dagger']
    # 
    vhs1 = np.zeros((vhs1_D.shape[0], vhs1_U0.shape[0], vhs1_U0.shape[0]))
    for i in range(vhs1_D.shape[0]):
        print(" vhs1_D[{}].shape :".format(i), vhs1_D[i].shape)
        vhs1[i] = vhs1_U0 @ vhs1_D[i] @ vhs1_Vdagger0
    # 
    vhs.append(vhs1)
    vhs.append(vhs1)
    print(f"{vhs1.shape = }")

    # time steps, 4 one body steps, 3 two body steps, ts_v is after sqrt
    ts_h = params['propagators_0']['ts_h']
    ts_v = params['propagators_0']['ts_v']
    ts_v_icf = np.zeros(( ts_v.shape[0] + 1),dtype=np.complex128)
    #
    ts_v_icf[0:1] = ts_v
    ts_v_icf[1] = 0.0
    print("ts_v_icf: ",ts_v_icf)
    print(f"{ts_h.shape = }")
    print(f"{ts_v.shape = }")
    #
    truncatedD = vhs1_U0.shape[1]
    #
    #####################################
    #first three slice for K+V
    #####################################
    choleskyNum = vhs[0].shape[0]
    #####################################
    choleskyVecMO = np.zeros((choleskyNum, L, L),dtype=np.complex64)
    for i in range(choleskyNum):
        choleskyVecMO[i] = vhs[0][i] * ts_v_icf[0]
    #####################################
    K = np.array(hmf[1] *  ts_h[1])
    #
    svdVecs=choleskyVecMO
    svdNumber=choleskyNum
    ############################
    #ATTention: there is a transpose between python and c++ i/o trans
    KT=np.zeros((L, L),dtype=np.complex128)
    KT[:L,:L]=K.transpose()

    svdVecsT=np.zeros((svdNumber, L, L),dtype=np.complex128)
    for i in range(svdNumber):
        svdVecsT[i,0:L,0:L]=svdVecs[i].transpose()

    ##########################
    KT_U=np.zeros((truncatedD, L),dtype=np.complex128)
    KT_U[0:truncatedD,0:L]=hmf1_U.transpose()
    # 
    KT_D=np.zeros((truncatedD, truncatedD),dtype=np.complex128)
    KT_D[0:truncatedD,0:truncatedD]=hmf1_D.transpose() *  ts_h[1]
    # 
    KT_Vdagger=np.zeros((L, truncatedD),dtype=np.complex128)
    KT_Vdagger[0:L,0:truncatedD]=hmf1_Vdagger.transpose()
    ##########################
    ##########################
    choleskyVecMO_matrix_U0T=np.zeros((truncatedD, L),dtype=np.complex128)
    choleskyVecMO_matrix_U0T[0:truncatedD,0:L]=vhs1_U0.transpose()
    #
    choleskyVecMO_matrix_DT=np.zeros((svdNumber, truncatedD, truncatedD),dtype=np.complex128)
    for i in range(svdNumber):
        choleskyVecMO_matrix_DT[i,0:truncatedD,0:truncatedD]=vhs1_D[i].transpose() * ts_v_icf[0]
    #
    choleskyVecMO_matrix_Vdagger0T=np.zeros((L, truncatedD),dtype=np.complex128)
    choleskyVecMO_matrix_Vdagger0T[0:L,0:truncatedD]=vhs1_Vdagger0.transpose()
    ##########################
    #
    f = h5py.File(name+"_"+str(0), "w")
    f.create_dataset("L",              (1,),                                data=[L],           dtype='int')
    f.create_dataset("SD2sL",              (1,),                                data=[L],           dtype='int')
    f.create_dataset("truncatedDup",              (1,),                       data=[truncatedD],           dtype='int')
    f.create_dataset("truncatedDdn",              (1,),                       data=[truncatedD],           dtype='int')
    f.create_dataset("Nup",              (1,),                                data=[Nup],           dtype='int')
    f.create_dataset("Ndn",              (1,),                                data=[Ndn],           dtype='int')
    f.create_dataset("N",            (1,),                                data=[Nup+Ndn],                     dtype='int')
    f.create_dataset("svdNumber", (1,),                                data=[svdNumber],           dtype='int')
    f.create_dataset("truncatedSvdNumber", (1,),                                data=[svdNumber],           dtype='int')
    f.create_dataset("K_r",              ((L)**2,),                 data=KT.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("K_U_r",              ((L)*(truncatedD),),                 data=KT_U.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("K_D_r",              ((truncatedD)*(truncatedD),),                 data=KT_D.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("K_Vdagger_r",              ((L)*(truncatedD),),                 data=KT_Vdagger.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_r",   (svdNumber*(L)**2,), data=svdVecsT.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdVecs_U0_r",   ((L)*(truncatedD),), data=choleskyVecMO_matrix_U0T.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdVecs_D_r",   (svdNumber*(truncatedD)**2,), data=choleskyVecMO_matrix_DT.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdVecs_Vdagger0_r",   ((L)*(truncatedD),), data=choleskyVecMO_matrix_Vdagger0T.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdBg_r",     (svdNumber,), data=np.zeros(svdNumber),   dtype='float64')
    f.create_dataset("K_i",              ((L)**2,),                 data=KT.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("K_U_i",              ((L)*(truncatedD),),                 data=KT_U.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("K_D_i",              ((truncatedD)*(truncatedD),),                 data=KT_D.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("K_Vdagger_i",              ((L)*(truncatedD),),                 data=KT_Vdagger.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_i",   (svdNumber*(L)**2,), data=svdVecsT.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_U0_i",   ((L)*(truncatedD),), data=choleskyVecMO_matrix_U0T.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdVecs_D_i",   (svdNumber*(truncatedD)**2,), data=choleskyVecMO_matrix_DT.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdVecs_Vdagger0_i",   ((L)*(truncatedD),), data=choleskyVecMO_matrix_Vdagger0T.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans  
    f.create_dataset("svdBg_i",     (svdNumber,),                  data=np.zeros(svdNumber),   dtype='float64')
    f.close()


##################################################################
# VAFQMC SD to AFQMC SD
##################################################################
def write_modelJastrowParam_lowRank_SD_from_checkpoint_lowRank_SD_slice1(hamiltonian_path="hamiltonian.pkl", param_path="checkpoint.pkl",  name="model_Jastrow_param"):
    #|\phiT> = \prod [exp(-tT)exp(-tV)]|\phi_0>
        
    # also 2.4, 2.7, 3.0, 3.6, 4.2
    hamil = load_pickle(hamiltonian_path)
    ckpt = load_pickle(param_path)
    params = ckpt[1][1]['params']['ansatz']

    # see https://github.com/y1xiaoc/hafqmc/blob/master/hafqmc/hamiltonian.py#L334
    # for detailed meaning of the arrays
    h1e, ceri, h1e_U, h1e_D, h1e_Vdagger, ceri_U, ceri_D, ceri_Vdagger, enuc, (wfn_a, wfn_b), aux = hamil
    # 
    halfL = h1e.shape[0]
    L = 2*h1e.shape[0]
    Nup = wfn_a.shape[1]
    Ndn = wfn_b.shape[1]
    N = Nup + Ndn
    # basis size 28
    # number of electrons 14 (7 up + 7 down)
    print('shape of the hamiltonian arrays:')
    print('one body operator K (h1e):', h1e.shape)
    print('two body operator after cholesky (ceri):', ceri.shape)
    print('nuclear contribution of energy (enuc):', enuc)
    print('spin up wfn coeffs (wfn_a):', wfn_a.shape)
    print('spin down wfn coeffs (wfn_b):', wfn_b.shape)
    print('ao2mo rotaton matrix ("orth_mat"):', aux['orth_mat'].shape)

    # remove extra layers
    # params = params['params']['ansatz']

    # one body operators at different slices, in GHF
    hmf=[]
    hmf0_U = params['propagators_0']['hmf_ops_0']['hmf_U']
    hmf0_D = params['propagators_0']['hmf_ops_0']['hmf_D']
    hmf0_Vdagger = params['propagators_0']['hmf_ops_0']['hmf_Vdagger']
    hmf0 = hmf0_U @ hmf0_D @ hmf0_Vdagger
    hmf.append(hmf0)
    hmf1_U = params['propagators_0']['hmf_ops_1']['hmf_U']
    hmf1_D = params['propagators_0']['hmf_ops_1']['hmf_D']
    hmf1_Vdagger = params['propagators_0']['hmf_ops_1']['hmf_Vdagger']
    hmf1 = hmf1_U @ hmf1_D @ hmf1_Vdagger
    hmf.append(hmf1)
    print(f"{hmf0.shape = }")

    # decompsed two body operators, same for all slices, 100 aux fields
    vhs=[]
    vhs1_U0 = params['propagators_0']['vhs_ops_0']['vhs_U0']
    vhs1_D = params['propagators_0']['vhs_ops_0']['vhs_D']
    vhs1_Vdagger0 = params['propagators_0']['vhs_ops_0']['vhs_V0dagger']
    # 
    vhs1 = np.zeros((vhs1_D.shape[0], vhs1_U0.shape[0], vhs1_U0.shape[0]))
    for i in range(vhs1_D.shape[0]):
        print(" vhs1_D[{}].shape :".format(i), vhs1_D[i].shape)
        vhs1[i] = vhs1_U0 @ vhs1_D[i] @ vhs1_Vdagger0
    # 
    vhs.append(vhs1)
    vhs.append(vhs1)
    print(f"{vhs1.shape = }")

    # time steps, 4 one body steps, 3 two body steps, ts_v is after sqrt
    ts_h = params['propagators_0']['ts_h']
    ts_v = params['propagators_0']['ts_v']
    ts_v_icf = np.zeros(( ts_v.shape[0] + 1),dtype=np.complex128)
    #
    ts_v_icf[0:1] = ts_v
    ts_v_icf[1] = 0.0
    print("ts_v_icf: ",ts_v_icf)
    print(f"{ts_h.shape = }")
    print(f"{ts_v.shape = }")
    #
    truncatedD = vhs1_U0.shape[1]
    #
    #####################################
    #first three slice for K+V
    #####################################
    choleskyNum = vhs[0].shape[0]
    #####################################
    choleskyVecMO = np.zeros((choleskyNum, L, L),dtype=np.complex64)
    for i in range(choleskyNum):
        choleskyVecMO[i] = vhs[0][i] * ts_v_icf[0]
    #####################################
    K = np.array(hmf[1] *  ts_h[1])
    #
    svdVecs=choleskyVecMO
    svdNumber=choleskyNum
    ############################
    #ATTention: there is a transpose between python and c++ i/o trans
    KT=np.zeros((L, L),dtype=np.complex128)
    KT[:L,:L]=K.transpose()

    svdVecsT=np.zeros((svdNumber, L, L),dtype=np.complex128)
    for i in range(svdNumber):
        svdVecsT[i,0:L,0:L]=svdVecs[i].transpose()

    ##########################
    KT_U=np.zeros((truncatedD, L),dtype=np.complex128)
    KT_U[0:truncatedD,0:L]=hmf1_U.transpose()
    # 
    KT_D=np.zeros((truncatedD, truncatedD),dtype=np.complex128)
    KT_D[0:truncatedD,0:truncatedD]=hmf1_D.transpose() *  ts_h[1]
    # 
    KT_Vdagger=np.zeros((L, truncatedD),dtype=np.complex128)
    KT_Vdagger[0:L,0:truncatedD]=hmf1_Vdagger.transpose()
    ##########################
    ##########################
    choleskyVecMO_matrix_U0T=np.zeros((truncatedD, L),dtype=np.complex128)
    choleskyVecMO_matrix_U0T[0:truncatedD,0:L]=vhs1_U0.transpose()
    #
    choleskyVecMO_matrix_DT=np.zeros((svdNumber, truncatedD, truncatedD),dtype=np.complex128)
    for i in range(svdNumber):
        choleskyVecMO_matrix_DT[i,0:truncatedD,0:truncatedD]=vhs1_D[i].transpose() * ts_v_icf[0]
    #
    choleskyVecMO_matrix_Vdagger0T=np.zeros((L, truncatedD),dtype=np.complex128)
    choleskyVecMO_matrix_Vdagger0T[0:L,0:truncatedD]=vhs1_Vdagger0.transpose()
    ##########################
    #
    f = h5py.File(name+"_"+str(0), "w")
    f.create_dataset("L",              (1,),                                data=[L],           dtype='int')
    f.create_dataset("SD2sL",              (1,),                                data=[halfL],           dtype='int')
    f.create_dataset("truncatedDup",              (1,),                       data=[truncatedD//2],           dtype='int')
    f.create_dataset("truncatedDdn",              (1,),                       data=[truncatedD//2],           dtype='int')
    f.create_dataset("Nup",              (1,),                                data=[Nup],           dtype='int')
    f.create_dataset("Ndn",              (1,),                                data=[Ndn],           dtype='int')
    f.create_dataset("N",            (1,),                                data=[Nup+Ndn],                     dtype='int')
    f.create_dataset("svdNumber", (1,),                                data=[svdNumber],           dtype='int')
    f.create_dataset("truncatedSvdNumber", (1,),                                data=[svdNumber],           dtype='int')
    f.create_dataset("K_r",              ((L)**2,),                 data=KT.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("K_U_r",              ((L)*(truncatedD),),                 data=KT_U.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("K_D_r",              ((truncatedD)*(truncatedD),),                 data=KT_D.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("K_Vdagger_r",              ((L)*(truncatedD),),                 data=KT_Vdagger.real.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_r",   (svdNumber*(L)**2,), data=svdVecsT.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdVecs_U0_r",   ((L)*(truncatedD),), data=choleskyVecMO_matrix_U0T.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdVecs_D_r",   (svdNumber*(truncatedD)**2,), data=choleskyVecMO_matrix_DT.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdVecs_Vdagger0_r",   ((L)*(truncatedD),), data=choleskyVecMO_matrix_Vdagger0T.real.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdBg_r",     (svdNumber,), data=np.zeros(svdNumber),   dtype='float64')
    f.create_dataset("K_i",              ((L)**2,),                 data=KT.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("K_U_i",              ((L)*(truncatedD),),                 data=KT_U.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("K_D_i",              ((truncatedD)*(truncatedD),),                 data=KT_D.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("K_Vdagger_i",              ((L)*(truncatedD),),                 data=KT_Vdagger.imag.ravel(),                 dtype='float64')    #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_i",   (svdNumber*(L)**2,), data=svdVecsT.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans
    f.create_dataset("svdVecs_U0_i",   ((L)*(truncatedD),), data=choleskyVecMO_matrix_U0T.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdVecs_D_i",   (svdNumber*(truncatedD)**2,), data=choleskyVecMO_matrix_DT.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans                    
    f.create_dataset("svdVecs_Vdagger0_i",   ((L)*(truncatedD),), data=choleskyVecMO_matrix_Vdagger0T.imag.ravel(),     dtype='float64')      #ATTention: there is a transpose between python and c++ i/o trans  
    f.create_dataset("svdBg_i",     (svdNumber,),                  data=np.zeros(svdNumber),   dtype='float64')
    f.close()


##################################################################
# Hamiltonian for model_Jastrow_param
##################################################################
# VAFQMC SD1s to AFQMC SD1s
##################################################################
def write_UHFSD2s_from_checkpoint_lowRank_SD1s_Kphi(hamiltonian_path="hamiltonian.pkl", param_path="ckpt_2000.pkl", JastrowExpM=[2], filename=None, noise=0.0):
        
    # also 2.4, 2.7, 3.0, 3.6, 4.2
    hamil = load_pickle(hamiltonian_path)
    ckpt = load_pickle(param_path)
    params = ckpt[1][1]['params']['ansatz']

    h1e, ceri, h1e_U, h1e_D, h1e_Vdagger, ceri_U, ceri_D, ceri_Vdagger, enuc, (wfn_a, wfn_b), aux = hamil
    # 
    L = h1e.shape[0]
    Nup = wfn_a.shape[1]
    Ndn = wfn_b.shape[1]
    #####################################
    hmf=[]
    hmf0_U = params['propagators_0']['hmf_ops_0']['hmf_U']
    hmf0_D = params['propagators_0']['hmf_ops_0']['hmf_D']
    hmf0_Vdagger = params['propagators_0']['hmf_ops_0']['hmf_Vdagger']
    hmf0 = hmf0_U @ hmf0_D @ hmf0_Vdagger
    hmf.append(hmf0)
    hmf1_U = params['propagators_0']['hmf_ops_1']['hmf_U']
    hmf1_D = params['propagators_0']['hmf_ops_1']['hmf_D']
    hmf1_Vdagger = params['propagators_0']['hmf_ops_1']['hmf_Vdagger']
    hmf1 = hmf1_U @ hmf1_D @ hmf1_Vdagger
    hmf.append(hmf1)
    print(f"{hmf0.shape = }")
    ts_h = params['propagators_0']['ts_h']
    ####
    K_temp = np.array(hmf[0] *  ts_h[0] * -1.0)
    #
    wfn0_a = np.array(params['wfn_a'])
    wfn0_b = np.array(params['wfn_b'])
    #
    exp_K = matrix_exponential(K_temp, JastrowExpM[0])
    k_wfn0_a = np.dot( exp_K, wfn0_a )
    k_wfn0_b = np.dot( exp_K, wfn0_b )
    #####################################
    # initial wavefunction in GHF to start projection

    f = open(filename, 'w')

    f.write('{:26.18e} {:26.18e} \n'.format(0.0,0.0))

    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(L,Nup))
    for i in range(Nup):
        for j in range(L):
                f.write( '{:26.18e} {:26.18e} \n'.format( k_wfn0_a[j,i].real+noise*random.random(),0.0 ) )
    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(L,Ndn))
    for i in range(Ndn):
        for j in range(L):
                f.write( '{:26.18e} {:26.18e} \n'.format( k_wfn0_b[j,i].real+noise*random.random(),0.0 ) )
    f.close()

##################################################################
# VAFQMC SD to AFQMC SD
##################################################################
def write_UHFSD_from_checkpoint_lowRank_SD_Kphi(hamiltonian_path="hamiltonian.pkl", param_path="ckpt_2000.pkl", JastrowExpM=[2], filename=None, noise=0.0):
        
    # also 2.4, 2.7, 3.0, 3.6, 4.2
    hamil = load_pickle(hamiltonian_path)
    ckpt = load_pickle(param_path)
    params = ckpt[1][1]['params']['ansatz']

    h1e, ceri, h1e_U, h1e_D, h1e_Vdagger, ceri_U, ceri_D, ceri_Vdagger, enuc, (wfn_a, wfn_b), aux = hamil
    # 
    halfL = h1e.shape[0]
    L = 2*h1e.shape[0]
    Nup = wfn_a.shape[1]
    Ndn = wfn_b.shape[1]
    N = Nup + Ndn
    #####################################
    hmf=[]
    hmf0_U = params['propagators_0']['hmf_ops_0']['hmf_U']
    hmf0_D = params['propagators_0']['hmf_ops_0']['hmf_D']
    hmf0_Vdagger = params['propagators_0']['hmf_ops_0']['hmf_Vdagger']
    hmf0 = hmf0_U @ hmf0_D @ hmf0_Vdagger
    hmf.append(hmf0)
    hmf1_U = params['propagators_0']['hmf_ops_1']['hmf_U']
    hmf1_D = params['propagators_0']['hmf_ops_1']['hmf_D']
    hmf1_Vdagger = params['propagators_0']['hmf_ops_1']['hmf_Vdagger']
    hmf1 = hmf1_U @ hmf1_D @ hmf1_Vdagger
    hmf.append(hmf1)
    print(f"{hmf0.shape = }")
    ts_h = params['propagators_0']['ts_h']
    ####
    K_temp = np.array(hmf[0] *  ts_h[0] * -1.0)
    #
    wfn0_a = np.array(params['wfn_a'])
    #
    exp_K = matrix_exponential(K_temp, JastrowExpM[0])
    k_wfn0_a = np.dot( exp_K, wfn0_a )
    #####################################
    # initial wavefunction in GHF to start projection

    f = open(filename, 'w')

    f.write('{:26.18e} {:26.18e} \n'.format(0.0,0.0))

    f.write('{:26d} \n'.format(2))
    f.write('{:26d} {:26d} \n'.format(L,N))
    for i in range(N):
        for j in range(L):
                f.write( '{:26.18e} {:26.18e} \n'.format( k_wfn0_a[j,i].real+noise*random.random(),0.0 ) )
    f.close()
##################################################################
