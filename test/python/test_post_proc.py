from itertools import product
from irbasis_x.freq import box, check_full_convention, from_ph_convention, to_ph_convention
from irbasis_x import atom
from alpscthyb.post_proc import VertexEvaluatorAtomED, legendre_to_tau, VertexEvaluatorU0, load_irbasis
from alpscthyb.interaction import hubbard_asymmU
import numpy as np
from scipy.special import eval_legendre
import pytest

def _atomic_F(U, beta, wsample_full):
    """ Compute full vertex of Hubbard atom"""
    wsample_full = check_full_convention(*wsample_full)
    wsample_ph = to_ph_convention(*wsample_full)
    nf = 2 
    Fuu_, Fud_ = atom.full_vertex_ph(U, beta, *wsample_ph)
    # Eq. (D4b) in PRB 86, 125114 (2012)
    Fbarud_ = - atom.full_vertex_ph(U, beta,
        wsample_ph[0],
        wsample_ph[0]+wsample_ph[2],
        wsample_ph[1]-wsample_ph[0])[1]
    Floc = np.zeros((len(wsample_ph[0]), nf, nf, nf, nf), dtype=np.complex128)
    Floc[:, 0, 0, 0, 0] = Floc[:, 1, 1, 1, 1] =  Fuu_
    Floc[:, 0, 0, 1, 1] = Floc[:, 1, 1, 0, 0] =  Fud_
    Floc[:, 1, 0, 0, 1] = Floc[:, 0, 1, 1, 0] =  Fbarud_
    return beta * Floc

def test_legendre_to_tau():
    beta = 1.25
    nl = 2
    gl = np.identity(nl, dtype=np.complex128)
    tau = np.array([0, 0.1*beta, 0.5*beta, beta])
    gtau = legendre_to_tau(gl, tau, beta)

    x = 2*tau/beta-1
    for l in range(nl):
        Pl = eval_legendre(l, x)
        gtau_ref = (np.sqrt(2*l+1.) * Pl/beta)
        np.testing.assert_allclose(gtau_ref, gtau[:,l])

def _mk_asymU(nflavors):
    asymU = \
        np.random.randn(nflavors, nflavors, nflavors, nflavors) + \
        1J * np.random.randn(nflavors, nflavors, nflavors, nflavors)
    asymU = 0.5*(asymU - asymU.transpose((2,1,0,3)))
    asymU = 0.5*(asymU - asymU.transpose((0,3,2,1)))
    asymU = 0.5*(asymU + asymU.transpose((1,0,3,2)).conj())
    return asymU

def _mk_Delta_l(beta, nflavors, basis_f, nbath=10):
    # Hybridization function
    V = np.random.randn(nbath,nflavors) + 1J*np.random.randn(nbath,nflavors)
    eb = np.linspace(-1,1,nbath)
    iv = 1J*basis_f.wsample*np.pi/beta
    Delta_w = np.einsum('bi,wb,bj->wij', V.conj(), 1/(iv[:,None]-eb[None,:]), V)
    Delta_l = basis_f.fit_iw(Delta_w.reshape((-1,nflavors**2))).\
        reshape((-1,nflavors,nflavors))
    return Delta_l

def _almost_equal(actural, dersired, rtol=1e-8, atol=1e-8):
    diff = np.abs(actural - dersired)
    return diff.max() < rtol * np.abs(dersired).max() + atol

@pytest.mark.parametrize("nflavors", [2, 3, 4])
def test_F_weak_coupling(nflavors):
    beta = 5.0
    Lambda = 100.0

    asymU = _mk_asymU(nflavors)

    basis_f = load_irbasis('F', Lambda, beta, 1e-10)
    basis_b = load_irbasis('B', Lambda, beta, 1e-10)

    hopping = np.random.randn(nflavors, nflavors) + 1J*np.random.randn(nflavors, nflavors) 
    hopping = hopping + hopping.T.conj()

    Delta_l = _mk_Delta_l(beta, nflavors, basis_f)

    evalU0 = VertexEvaluatorU0(nflavors, beta, basis_f, basis_b, hopping, Delta_l, asymU=asymU)

    wsample_full = box(4, 3, return_conv='full')
    F = evalU0.compute_F(wsample_full)

    # swap: 1<->2
    F_swap13 = evalU0.compute_F(
        (wsample_full[2], wsample_full[1], wsample_full[0], wsample_full[3]))
    assert _almost_equal(F, -F_swap13.transpose((0,3,2,1,4)))

    # swap: 2<->4
    F_swap24 = evalU0.compute_F(
        (wsample_full[0], wsample_full[3], wsample_full[2], wsample_full[1]))
    assert _almost_equal(F, -F_swap24.transpose((0,1,4,3,2)))

    # complex conjugate
    F_minus = evalU0.compute_F((-wsample_full[3], -wsample_full[2], -wsample_full[1], -wsample_full[0]))
    assert _almost_equal(F.conj(), F_minus.transpose((0,4,3,2,1)))


@pytest.mark.parametrize("nflavors", [1, 2, 3])
def test_F_U0(nflavors):
    np.random.seed(100)
    beta = 1.5
    Lambda = 100.0
    basis_f = load_irbasis('F', Lambda, beta, 1e-10)
    basis_b = load_irbasis('B', Lambda, beta, 1e-10)

    asymU = np.zeros((nflavors,)*4)

    hopping = \
        np.random.randn(nflavors, nflavors) + 1J*np.random.randn(nflavors, nflavors)
    hopping = hopping + hopping.T.conj()

    # Hybridization function
    Delta_l = _mk_Delta_l(beta, nflavors, basis_f)

    evalU0 = VertexEvaluatorU0(nflavors, beta, basis_f, basis_b, hopping, Delta_l, asymU=asymU)

    wsample_ph = box(4,3, return_conv="ph")
    wsample_full = from_ph_convention(*wsample_ph)
    F = evalU0.compute_F(wsample_full)
    assert np.abs(F).max() < 1e-10


def test_Hubbard_atom():
    nflavors = 2
    beta = 1.0
    U = 1.0
    #beta = 5.0
    #U = 1.5

    asymU = hubbard_asymmU(U)

    mu = 0.5*U
    #hopping = -mu * np.identity(nflavors)
    hopping = np.random.randn(nflavors, nflavors)
    hopping = hopping + hopping.T.conj()

    evalatom = VertexEvaluatorAtomED(nflavors, beta, hopping, asymU)

    wsample_full = box(4, 5, return_conv='full')
    F_ed = evalatom.compute_F(wsample_full, True)

    #F_ref = _atomic_F(U, beta, wsample_full)
    #assert np.abs(F_ed-F_ref).max()/np.abs(F_ref).max() < 1e-8

    # Construct G^{v1,v2,v3,v4} from F
    v1, v2, v3, v4 = wsample_full
    g1 = evalatom.compute_giv(v1)
    g2 = evalatom.compute_giv(v2)
    g3 = evalatom.compute_giv(v3)
    g4 = evalatom.compute_giv(v4)
    g4pt_ref = (beta**2) * (
        np.einsum('w,wab,wcd->wabcd', v1==v2, g1, g3, optimize=True) -
        np.einsum('w,wad,wcb->wabcd', v1==v4, g1, g3, optimize=True)
    ) - np.einsum('waA,wBb,wcC,wDd,wABCD->wabcd', g1, g2, g3, g4, F_ed, optimize=True)

    g4pt_ed = evalatom.compute_g4pt(wsample_full)
    print(g4pt_ed)
    print(g4pt_ref)
    assert np.abs(g4pt_ed-g4pt_ref).max()/np.abs(g4pt_ref).max() < 1e-8