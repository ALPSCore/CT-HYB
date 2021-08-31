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
    beta = 1.5
    nflavors = 2
    mu = 0.1

    hopping = -mu * np.identity(nflavors)

    _, cdag_ops = construct_cdagger_ops(nflavors)
    c_ops = [op.transpose(copy=True) for op in cdag_ops]
    ham = construct_ham(
        hopping, np.zeros(4*(nflavors,)),
        cdag_ops)

    evals, evecs = np.linalg.eigh(ham.toarray())

    # Reference data
    Lambda = 1e+4
    basis_f = load_irbasis('F', Lambda, beta, 1e-15)
    basis_b = load_irbasis('B', Lambda, beta, 1e-15)
    evalU0 = VertexEvaluatorU0(nflavors, beta, basis_f, basis_b, 
        hopping, np.zeros((basis_f.dim(), nflavors, nflavors)))

    # fermionic sampling frequencies
    wfs = 2*np.arange(-10,10)+1

    # up, up
    g_uu_iv = compute_fermionic_2pt_corr_func(
        cdag_ops[0].transpose(),
        cdag_ops[0], beta, wfs, evals, evecs)
    iv = 1J * wfs * np.pi/beta
    ref = 1/(iv + mu)
    np.testing.assert_allclose(g_uu_iv, ref)

    # eta(v, w)_{uudd}
    #wsample_fb = box_fb(2, 3)
    wsample_fb = np.array([1]), np.array([2])
    q_ops = [c@ham-ham@c for c in c_ops]
    qdag_ops = [ham@cdag-cdag@ham for cdag in cdag_ops]
    
    print(evalU0.compute_eta(*wsample_fb).shape)
    print(evalU0.compute_eta(*wsample_fb)[:,0,0,1,1])
    eta_ref = evalU0.compute_eta(*wsample_fb)
    for i,j,k,l in product(range(nflavors), repeat=4):
        eta = compute_3pt_corr_func(
            q_ops[i], qdag_ops[j], cdag_ops[k]@c_ops[l],
            beta, wsample_fb, evals, evecs)
        print(i, j, k, l, eta_ref[:, i, j, k, l], eta)
        #print(np.abs(evalU0.compute_eta(*wsample_fb)).max())
