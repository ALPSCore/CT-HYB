import numpy as np
from numpy.lib.npyio import load
from alpscthyb.occupation_basis import *
from alpscthyb.exact_diag import *
from alpscthyb.interaction import *
from alpscthyb.post_proc import VertexEvaluatorU0, load_irbasis, VertexEvaluatorED, float_to_complex_array, reconst_vartheta
from irbasis_x.freq import box
from itertools import product
import pytest
import h5py

# Fermion-boson frequency box
def box_fb(nf, nb):
    wf = 2*np.arange(-nf,nf)+1
    wb = 2*np.arange(-nb,nb)
    v, w = np.broadcast_arrays(wf[:,None], wb[None,:])
    return v.ravel(), w.ravel()


def test_single_orb_Hubbard_atom():
    """ Hubbard atom at half filling"""
    U = 1.0
    mu = 0.5*U
    beta = 1.5
    nflavors = 2

    asymmU = check_asymm(hubbard_asymmU(U))
    _, cdag_ops = construct_cdagger_ops(nflavors)

    hopping = -mu * np.identity(nflavors)
    ham = construct_ham(hopping, asymmU, cdag_ops)

    evals, evecs = np.linalg.eigh(ham.toarray())

    # fermionic sampling frequencies
    wfs = 2*np.arange(-10,10)+1

    # up, up
    g_uu_iv = compute_fermionic_2pt_corr_func(
        cdag_ops[0].transpose(),
        cdag_ops[0], beta, wfs, evals, evecs)

    iv = 1J * wfs * np.pi/beta
    ref = 0.5/(iv - 0.5*U) + 0.5/(iv + 0.5*U)
    
    np.testing.assert_allclose(g_uu_iv, ref)


def test_single_orb_Hubbard_atom_U0():
    """ Hubbard atom at half filling and U=0"""
    beta = 2.0
    nflavors = 2
    mu = 0.1

    hopping = -mu * np.identity(nflavors)

    _, cdag_ops = construct_cdagger_ops(nflavors)
    c_ops = [op.transpose(copy=True) for op in cdag_ops]
    n_ops = [cdag_ops[i]@c_ops[i] for i in range(nflavors)]
    ham = construct_ham(
        hopping, np.zeros(4*(nflavors,)),
        cdag_ops)
    q_ops = [c@ham-ham@c for c in c_ops]
    qdag_ops = [ham@cdag-cdag@ham for cdag in cdag_ops]

    evals, evecs = np.linalg.eigh(ham.toarray())

    # Reference data
    Lambda = 1e+4
    evalU0 = VertexEvaluatorU0(nflavors, beta, hopping, Lambda=Lambda)

    # Sampling frequencies
    wfs = 2*np.arange(-10,10)+1
    wbs = 2*np.arange(-10,10)

    # G_{uu}(iv)
    g_uu_iv = compute_fermionic_2pt_corr_func(
        cdag_ops[0].transpose(),
        cdag_ops[0], beta, wfs, evals, evecs)
    iv = 1J * wfs * np.pi/beta
    ref = 1/(iv + mu)
    np.testing.assert_allclose(g_uu_iv, ref)

    # int_0^\beta d tau e^{iw tau) <T n_{ab}(tau) n_{cd}(0)>
    lambda_iw_ref = evalU0.compute_lambda(wbs)
    lambda_iw = np.zeros_like(lambda_iw_ref)
    for i, j, k, l in product(range(nflavors),repeat=4):
        lambda_iw[:,i,j,k,l] = \
            compute_bosonic_2pt_corr_func(
                cdag_ops[i]@c_ops[j],
                cdag_ops[k]@c_ops[l],
                beta, wbs, evals, evecs)
    assert np.abs(lambda_iw_ref-lambda_iw).max() < 1e-8

    """
    # eta(v, w)_{uudd}
    wsample_fb = box_fb(2, 3)
    
    eta_ref = evalU0.compute_eta(*wsample_fb)
    for i,j,k,l in product(range(nflavors), repeat=4):
        eta = compute_3pt_corr_func(
            q_ops[i], qdag_ops[j], cdag_ops[k]@c_ops[l],
            beta, wsample_fb, evals, evecs)
        assert all(np.abs(eta_ref[:, i, j, k, l]-eta) < 1e-8)

    # h(v1, v2, v3, v4)
    wsample_full = box(2,3, return_conv='full')
    h_ref = evalU0.compute_h(wsample_full)
    h_ed = np.zeros_like(h_ref)
    for i, j, k, l in product(range(nflavors),repeat=4):
        h_ed[:,i,j,k,l] = compute_4pt_corr_func(
            q_ops[i], qdag_ops[j], q_ops[k], qdag_ops[l],
            beta, wsample_full, evals, evecs
        )
    assert np.abs(h_ed-h_ref).max() < 1e-8
    """

def _to_spin_full(mat):
    assert mat.ndim == 2
    N1, N2 = mat.shape[0], mat.shape[1]
    mat_full = np.zeros((N1,2,N2,2), dtype=mat.dtype)
    mat_full[:,0,:,0] = mat
    mat_full[:,1,:,1] = mat
    return mat_full.reshape((N1*2, N2*2))


def test_Dimer():
    """
    https://github.com/TRIQS/benchmarks.git: Dimer
    """
    norb_imp = 2
    beta = 5.                       # Inverse temperature
    mu = 0.0                        # Chemical potential
    eps = np.array([0.0, 0.1])         # Impurity site energies
    t = 0.2
    
    eps_bath = np.array([0.27, -0.4])  # Bath site energies
    t_bath = 0.0    
    
    U = 1.0                          # On-site interaction
    J = 0.2                         # Hunds coupling
    
    # Impurity (local) hamiltonian
    hopping_imp = _to_spin_full(np.diag(eps - mu) - np.matrix([[0, t], [t, 0]]))
    asymU_imp = slater_kanamori_asymm(2, U, J)
    asymU_imp = check_asymm(asymU_imp)
    
    hopping_bath = _to_spin_full(np.diag(eps_bath) - np.matrix([[0, t_bath], [t_bath, 0]]))
    hopping_coup = _to_spin_full(np.ones((2,2)))
    ed = VertexEvaluatorED(beta, hopping_imp, asymU_imp, hopping_bath, hopping_coup)
    ed0 = VertexEvaluatorED(beta, hopping_imp, np.zeros_like(asymU_imp),
        hopping_bath, hopping_coup)
    assert np.abs(ed.evals_imp[0] - (-0.156155)) < 1e-5

    # ED data
    with h5py.File('dimer_pyed.h5', 'r') as h5:
        #nsp = 2
        gup = float_to_complex_array(h5['/G/up/data'][()])
        gdn = float_to_complex_array(h5['/G/dn/data'][()])
        giv_ref = np.zeros((gup.shape[0], norb_imp, 2, norb_imp, 2), dtype=np.complex128)
        giv_ref[:, :, 0, :, 0] = gup
        giv_ref[:, :, 1, :, 1] = gdn
        giv_ref = giv_ref.reshape((gup.shape[0], 2*norb_imp, 2*norb_imp))
    
        dens_mat_ref = np.zeros((norb_imp, 2, norb_imp, 2), dtype=np.complex128)
        dens_mat_ref[:, 0, :, 0] = float_to_complex_array(h5['/dens_mat/up'][()])
        dens_mat_ref[:, 1, :, 1] = float_to_complex_array(h5['/dens_mat/dn'][()])
        dens_mat_ref = dens_mat_ref.reshape((2*norb_imp, 2*norb_imp))
        #gs_ene_ref = h5['ground_state_energy'][()]

    # Fermionic sampling frequencies
    wfs = 2*np.arange(-giv_ref.shape[0]//2, giv_ref.shape[0]//2) + 1

    assert (np.abs(ed.get_dm() - dens_mat_ref) < 1e-8).all()

    giv = ed.compute_giv(wfs)
    #sigma_iv = ed.compute_(giv, wfs)
    assert np.abs(giv-giv_ref).max() < 1e-8

    vartheta = ed.compute_vartheta(wfs)
    #print(giv[:4,0,0])
    #print(giv_ref[:4,0,0])

    g0iv = ed0.compute_giv(wfs)
    g0iv_from_delta = ed0.compute_g0iv(wfs)
    assert np.abs(g0iv-g0iv_from_delta).max() < 1e-8
    vartheta_reconst = reconst_vartheta(ed.get_asymU(), giv, g0iv_from_delta, ed.get_dm())

    for i in range(wfs.size):
        print(wfs[i],
            vartheta[i,0,0].real, vartheta[i,0,0].imag,
            vartheta_reconst[i,0,0].real, vartheta_reconst[i,0,0].imag
            )