import numpy as np
from numpy.lib.npyio import load
from alpscthyb.occupation_basis import *
from alpscthyb.exact_diag import *
from alpscthyb.interaction import *
from alpscthyb.post_proc import VertexEvaluatorU0, load_irbasis
from irbasis_x.freq import box
from itertools import product
import pytest

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
    basis_f = load_irbasis('F', Lambda, beta, 1e-15)
    basis_b = load_irbasis('B', Lambda, beta, 1e-15)
    evalU0 = VertexEvaluatorU0(nflavors, beta, basis_f, basis_b, 
        hopping, np.zeros((basis_f.dim(), nflavors, nflavors)))

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
    print(h_ref[:,0,0,0,0])
    print(h_ed[:,0,0,0,0])
    assert np.abs(h_ed-h_ref).max() < 1e-8