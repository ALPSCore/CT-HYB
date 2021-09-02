from itertools import product
from irbasis_x.freq import box, check_full_convention, from_ph_convention, to_ph_convention
from irbasis_x import atom
from alpscthyb.post_proc import VertexEvaluatorAtomED, legendre_to_tau, VertexEvaluatorU0, load_irbasis
from alpscthyb.interaction import hubbard_asymmU
import numpy as np
from scipy.special import eval_legendre
from scipy.linalg import expm
import pytest

def _einsum(subscripts, *operands):
    return np.einsum(subscripts, *operands, optimize=True)

def _rotate_system(rotmat, hopping, Delta_l, asymU):
    hopping_rot = _mk_hermite(rotmat.conj().T@hopping@rotmat)
    Delta_l_rot = _einsum('ij,Ljk,kl->Lil', rotmat.conj().T, Delta_l, rotmat)
    asymU_rot = _einsum('Aa,Bb,Cc,Dd,ACBD->acbd',
        rotmat.conj(), rotmat, rotmat.conj(), rotmat, asymU)
    return hopping_rot, Delta_l_rot, asymU_rot


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

def _mk_hermite(mat):
    return 0.5*(mat + mat.T.conj())

def _mk_Delta_l(beta, nflavors, basis_f, nbath=10):
    # Hybridization function
    V = np.random.randn(nbath,nflavors) + 1J*np.random.randn(nbath,nflavors)
    eb = np.linspace(-1,1,nbath)
    iv = 1J*basis_f.wsample*np.pi/beta
    Delta_w = _einsum('bi,wb,bj->wij', V.conj(), 1/(iv[:,None]-eb[None,:]), V)
    Delta_l = basis_f.fit_iw(Delta_w.reshape((-1,nflavors**2))).\
        reshape((-1,nflavors,nflavors))
    return Delta_l

def _mk_rnd_umat(N):
    hmat = np.random.randn(N,N) + 1J*np.random.randn(N,N)
    hmat = hmat + hmat.conj().T
    return expm(1J*hmat)

def _almost_equal(actural, dersired, rtol=1e-8, atol=1e-8):
    diff = np.abs(actural - dersired)
    print("acutual", actural)
    print("desired", dersired)
    print(diff.max(), rtol * np.abs(dersired).max(), atol)
    return diff.max() < rtol * np.abs(dersired).max() + atol

@pytest.mark.parametrize("nflavors", [2, 3, 4])
def test_F_weak_coupling(nflavors):
    beta = 5.0
    Lambda = 100.0

    asymU = _mk_asymU(nflavors)

    basis_f = load_irbasis('F', Lambda, beta, 1e-10)
    basis_b = load_irbasis('B', Lambda, beta, 1e-10)
    wfs = np.array([1,11,101,-1,-11,-101])

    hopping = _mk_hermite(np.random.randn(nflavors, nflavors) + 1J*np.random.randn(nflavors, nflavors))
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

    # Rotation in spin-orbital space
    rotmat = _mk_rnd_umat(nflavors)
    hopping_rot, Delta_l_rot, asymU_rot = _rotate_system(rotmat, hopping, Delta_l, asymU)
    evalU0_rot = VertexEvaluatorU0(nflavors, beta, basis_f, basis_b,
        hopping_rot, Delta_l_rot, asymU=asymU_rot)
    
    # Rotation of one-particle GF
    _almost_equal(
        evalU0_rot.compute_giv(wfs),
        _einsum('Aa,Bb,wAB->wab', rotmat.conj(), rotmat, evalU0.compute_giv(wfs))
    )

    # Rotation of full vertex
    F_rot = _einsum('Aa,Bb,Cc,Dd,wACBD->wacbd',
        rotmat.conj(), rotmat, rotmat.conj(), rotmat, F)
    _almost_equal(evalU0_rot.compute_F(wsample_full), F_rot)


@pytest.mark.parametrize("nflavors", [1, 2, 3])
def test_F_U0(nflavors):
    np.random.seed(100)
    beta = 1.5
    Lambda = 100.0
    basis_f = load_irbasis('F', Lambda, beta, 1e-10)
    basis_b = load_irbasis('B', Lambda, beta, 1e-10)

    asymU = np.zeros((nflavors,)*4)

    hopping = _mk_hermite(np.random.randn(nflavors, nflavors) + 1J*np.random.randn(nflavors, nflavors))

    # Hybridization function
    Delta_l = _mk_Delta_l(beta, nflavors, basis_f)

    evalU0 = VertexEvaluatorU0(nflavors, beta, basis_f, basis_b, hopping, Delta_l, asymU=asymU)

    wsample_ph = box(4,3, return_conv="ph")
    wsample_full = from_ph_convention(*wsample_ph)
    F = evalU0.compute_F(wsample_full)
    assert np.abs(F).max() < 1e-10


def test_Hubbard_atom():
    np.random.seed(100)
    nflavors = 2
    beta = 10.0
    U = 5.0

    asymU = hubbard_asymmU(U)

    mu = 0.5*U
    h = 0.1
    hopping = np.diag(np.array([h-mu,-h-mu]))

    evalatom = VertexEvaluatorAtomED(nflavors, beta, hopping, asymU)

    # First 10 matsubara freqs
    giv = evalatom.compute_giv(2*np.arange(10)+1)
    print("giv on first 10 freqs: (u,u): ", giv[:,0,0])
    print("giv on first 10 freqs: (d,d): ", giv[:,1,1])

    # From pomerol
    giv_uu_ref = np.array([
        -0.285062+1J*(-0.0467365),
        -0.256392+1J*(-0.125437),
        -0.213409+1J*(-0.172698),
        -0.170487+1J*(-0.191776),
        -0.134413+1J*(-0.193294),
        -0.106288+1J*(-0.186022),
        -0.0849517+1J*(-0.175163),
        -0.0688294+1J*(-0.163377),
        -0.0565607+1J*(-0.151893),
        -0.0471127+1J*(-0.14122)])
    _almost_equal(giv[:,0,0], giv_uu_ref)

    #wsample_full = box(4, 5, return_conv='full')
    wsample_full = \
        np.array([1,1]), \
        np.array([1,1]), \
        np.array([1,3]), \
        np.array([1,3])
    g4pt_pomerol = -beta * \
        np.array([3.5050141435761, 1.2829547341506-1j*0.806476785417485]) # Where is this sign come from?
    F_ed = evalatom.compute_F(wsample_full)
    #scrF_ed = evalatom._compute_scrF(wsample_full)

    # Construct G^{v1,v2,v3,v4} from F
    v1, v2, v3, v4 = wsample_full
    g1 = evalatom.compute_giv(v1)
    g2 = evalatom.compute_giv(v2)
    g3 = evalatom.compute_giv(v3)
    g4 = evalatom.compute_giv(v4)
    #calg1 = evalatom.compute_calgiv(v1)
    #calg2 = evalatom.compute_calgiv(v2)
    #calg3 = evalatom.compute_calgiv(v3)
    #calg4 = evalatom.compute_calgiv(v4)
    g4pt_ref = (beta**2) * (
        _einsum('w,wab,wcd->wabcd', v1==v2, g1, g3)-_einsum('w,wad,wcb->wabcd', v1==v4, g1, g3)
    ) -_einsum('waA,wBb,wcC,wDd,wABCD->wabcd', g1, g2, g3, g4, F_ed)
    #g4pt_ref = (beta**2) * (
        #_einsum('w,wab,wcd->wabcd', v1==v2, g1, g3) -_einsum('w,wad,wcb->wabcd', v1==v4, g1, g3)
    #) -_einsum('waA,wBb,wcC,wDd,wABCD->wabcd', calg1, calg2, calg3, calg4, scrF_ed)
#
    g4pt_ed = evalatom.compute_g4pt(wsample_full)
    #print(g4pt_ed)
    #print(g4pt_ref)
    #assert np.abs(g4pt_ed-g4pt_ref).max()/np.abs(g4pt_ref).max() < 1e-8
    #print("G4pt")
    #for i, j, k, l in product(range(nflavors), repeat=4):
        #print(i, j, k, l,
           #g4pt_ed[:,i,j,k,l],
           #g4pt_ref[:,i,j,k,l],
           #np.abs(g4pt_ed[:,i,j,k,l]- g4pt_ref[:,i,j,k,l])
        #)
    #print("")
    #print("F")
    #for i, j, k, l in product(range(nflavors), repeat=4):
        #print(i, j, k, l, F_ed[:,i,j,k,l])

    assert _almost_equal(g4pt_ed[:,0,0,1,1], g4pt_pomerol)
    assert _almost_equal(g4pt_ref[:,0,0,1,1], g4pt_pomerol, rtol=1e-5)
    #assert _almost_equal(g4pt_ed, g4pt_ref)

"""
def test_Hubbard_atom_debug():
    np.random.seed(100)
    nflavors = 2
    beta = 10.0
    U = 5.0

    asymU = hubbard_asymmU(U)

    mu = 0.5*U
    h = 0.1
    hopping = np.diag(np.array([h-mu,-h-mu]))

    evalatom = VertexEvaluatorAtomED(nflavors, beta, hopping, asymU)

    # First 10 matsubara freqs
    giv = evalatom.compute_giv(2*np.arange(10)+1)
    print("giv on first 10 freqs: (u,u): ", giv[:,0,0])
    print("giv on first 10 freqs: (d,d): ", giv[:,1,1])

    print("vab", evalatom.compute_v())

    # From pomerol
    wsample_full = \
        np.array([1,1]), \
        np.array([1,1]), \
        np.array([1,3]), \
        np.array([1,3])
    g4pt_pomerol = -beta * \
        np.array([3.5050141435761, 1.2829547341506-1j*0.806476785417485]) # Where is this sign come from?
    F_ed = evalatom.compute_F(wsample_full)
    scrF_ed = evalatom._compute_scrF(wsample_full)

    # Construct G^{v1,v2,v3,v4} from F
    v1, v2, v3, v4 = wsample_full
    g1 = evalatom.compute_giv(v1)
    g2 = evalatom.compute_giv(v2)
    g3 = evalatom.compute_giv(v3)
    g4 = evalatom.compute_giv(v4)
    calg1 = evalatom.compute_calgiv(v1)
    calg2 = evalatom.compute_calgiv(v2)
    calg3 = evalatom.compute_calgiv(v3)
    calg4 = evalatom.compute_calgiv(v4)
    #g4pt_ref = (beta**2) * (
        #_einsum('w,wab,wcd->wabcd', v1==v2, g1, g3)-_einsum('w,wad,wcb->wabcd', v1==v4, g1, g3)
    #) -_einsum('waA,wBb,wcC,wDd,wABCD->wabcd', g1, g2, g3, g4, F_ed)
    g4pt_ref = (beta**2) * (
        _einsum('w,wab,wcd->wabcd', v1==v2, g1, g3) -_einsum('w,wad,wcb->wabcd', v1==v4, g1, g3)
    ) -_einsum('waA,wBb,wcC,wDd,wABCD->wabcd', calg1, calg2, calg3, calg4, scrF_ed)

    g4pt_ed = evalatom.compute_g4pt(wsample_full)
    #print(g4pt_ed)
    #print(g4pt_ref)
    #assert np.abs(g4pt_ed-g4pt_ref).max()/np.abs(g4pt_ref).max() < 1e-8
    print("G4pt")
    for i, j, k, l in product(range(nflavors), repeat=4):
        print(i, j, k, l,
           g4pt_ed[:,i,j,k,l],
           g4pt_ref[:,i,j,k,l],
           np.abs(g4pt_ed[:,i,j,k,l]- g4pt_ref[:,i,j,k,l])
        )

    print("")
    print("F")
    for i, j, k, l in product(range(nflavors), repeat=4):
        print(i, j, k, l, F_ed[:,i,j,k,l])
    #assert _almost_equal(g4pt_ed[:,0,0,1,1], g4pt_pomerol) assert _almost_equal(g4pt_ref[:,0,0,1,1], g4pt_pomerol)
    #assert _almost_equal(g4pt_ed, g4pt_ref)
"""