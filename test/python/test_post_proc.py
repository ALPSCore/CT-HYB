from h5py._hl import dataset
import numpy as np
from irbasis_x.freq import box, from_ph_convention
from alpscthyb.post_proc import *
from alpscthyb.interaction import hubbard_asymmU
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

def _almost_equal(actual, desired, rtol=1e-8, atol=1e-8):
    diff = np.abs(actual - desired)
    return diff.max() < rtol * np.abs(desired).max() + atol

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

    wsample_full = \
        np.array([1,1]), \
        np.array([1,1]), \
        np.array([1,3]), \
        np.array([1,3])
    g4pt_pomerol = -beta * \
        np.array([3.5050141435761, 1.2829547341506-1j*0.806476785417485]) # Where is this sign come from?
    F_ed = evalatom.compute_F(wsample_full)

    # Construct G^{v1,v2,v3,v4} from F
    g4pt_reconst = evalatom.compute_g4pt(wsample_full, F=F_ed)
    g4pt_ed = evalatom.compute_g4pt_direct(wsample_full)

    assert _almost_equal(g4pt_ed[:,0,0,1,1], g4pt_pomerol)
    assert _almost_equal(g4pt_reconst[:,0,0,1,1], g4pt_pomerol, rtol=1e-5)


def test_three_orb_SOI_U0():
    nflavors = 6
    beta = 20.0
    Lambda = 1e+5
    basis_f = load_irbasis('F', Lambda, beta, 1e-10)
    basis_b = load_irbasis('B', Lambda, beta, 1e-10)

    soi = 1.0
    hopping = soi * (-0.5)*np.array( [
        [ 0,  0, -1J,  0,  0,  1], 
        [ 0,  0,  0,  1J, -1,  0], 
        [1J,  0,  0,   0,  0,-1J],
        [ 0,-1J,  0,   0,-1J,  0], 
        [ 0, -1,  0,  1J,  0,  0], 
        [ 1,  0, 1J,   0,  0,  0]], dtype=np.complex128)
    Delta_l = np.zeros((basis_f.dim(), nflavors, nflavors), dtype=np.complex128)
    evalU0 = VertexEvaluatorU0(nflavors, beta, basis_f, basis_b, hopping, Delta_l)

    wsample_full = box(4, 3, return_conv='full')
    F = evalU0.compute_F(wsample_full)
    assert np.abs(F).max() < 1e-8


def test_ft_three_point_obj():
    np.random.seed(100)
    beta = 10.0
    nflavors = 2

    #nconfig = 100
    #nfreqs = 10
    freq_max = 10
    nconfig = int(1E+4)
    nfreqs = int(1E+4)

    taus = []
    for _ in range(3):
        taus.append(beta * np.random.rand(nconfig))

    flavors = []
    for _ in range(4):
        flavors.append(np.random.randint(nflavors, size=nconfig))
    
    values = np.random.randn(nconfig) + 1J*np.random.randn(nconfig)

    worm_config_record = WormConfigRecord(
        [{'taus': taus, 'flavors': flavors, 'values': values}])

    wfs = 2*np.random.randint(freq_max, size=nfreqs) + 1
    wbs = 2*np.random.randint(freq_max, size=nfreqs)

    t1 = time.time()
    res_ref = ft_three_point_obj_ref(worm_config_record, (wfs, wbs), nflavors, beta)
    t2 = time.time()

    res = ft_three_point_obj(worm_config_record, (wfs, wbs), nflavors, beta)
    t3 = time.time()
    
    print(t2-t1, t3-t2)
    np.testing.assert_allclose(res, res_ref)


def _eval_exp(beta, wf, taus, sign):
    """ Exvaluate exp(1J*sign*PI*wf*taus/beta)"""
    wf_unique, wf_where = np.unique(wf, return_inverse=True)
    coeff = sign * 1J * np.pi/beta
    exp_unique = np.exp(coeff * wf_unique[:,None] * taus[None,:])
    return exp_unique[wf_where, :]


def ft_four_point_obj_ref(worm_config_record, wsample, nflavors, beta):
    wf1, wf2, wf3, wf4 = check_full_convention(*wsample)
    res = np.zeros((wf1.size,) + 4*(nflavors,), dtype=np.complex128)
    ndata = 0
    for dset in worm_config_record.datasets:
        ndata += dset['flavors'][0].size
        for f1, f2, f3, f4 in product(range(nflavors), repeat=4):
            flavors_data = dset['flavors']
            where = \
                np.logical_and(
                    np.logical_and(flavors_data[0] == f1, flavors_data[1] == f2),
                    np.logical_and(flavors_data[2] == f3, flavors_data[3] == f4)
                )
            res_ = np.ones((wf1.size, dset['taus'][0][where].size), dtype=np.complex128)
            res_ *= _eval_exp(beta, wf1, dset['taus'][0][where],  1)
            res_ *= _eval_exp(beta, wf2, dset['taus'][1][where], -1)
            res_ *= _eval_exp(beta, wf3, dset['taus'][2][where],  1)
            res_ *= _eval_exp(beta, wf4, dset['taus'][3][where], -1)
            res_ *= dset['values'][None, where]
            res[:,f1,f2,f3,f4] += np.sum(res_, axis=1)
    res = mpi.allreduce(res)
    ndata = mpi.allreduce(ndata)
    return res/ndata

def ft_three_point_obj_ref(worm_config_record, wsample, nflavors, beta):
    wfs, wbs = wsample
    wfs = check_fermionic(wfs)
    wbs = check_bosonic(wbs)
    res = np.zeros((wfs.size,) + 4*(nflavors,), dtype=np.complex128)
    ndata = 0
    for dset in worm_config_record.datasets:
        taus_f = dset['taus'][0] - dset['taus'][1]
        taus_b = dset['taus'][1] - dset['taus'][2]
        exp_ = np.exp(1J * np.pi * (
            wfs[:,None]*taus_f[None,:]+
            wbs[:,None]*taus_b[None,:])/beta)
        res_ = exp_ * dset['values'][None,:]
        flavors = dset['flavors']
        ndata += flavors[0].size
        for iconfig in range(flavors[0].size):
            res[:, flavors[0][iconfig],
                   flavors[1][iconfig],
                   flavors[2][iconfig],
                   flavors[3][iconfig],
            ] += res_[:, iconfig]
    res = mpi.allreduce(res)
    ndata = mpi.allreduce(ndata)
    return res/(ndata*beta)


def test_ft_four_point_obj():
    np.random.seed(100)
    beta = 10.0
    nflavors = 3

    nconfig = 100
    nfreqs = 10
    freq_max = 10

    nconfig = int(1E+4)
    nfreqs = int(1E+4)

    taus = []
    for _ in range(4):
        taus.append(beta * np.random.rand(nconfig))

    flavors = []
    for _ in range(4):
        flavors.append(np.random.randint(nflavors-1, size=nconfig))
    
    values = np.random.randn(nconfig) + 1J*np.random.randn(nconfig)

    worm_config_record = WormConfigRecord(
        [{'taus': taus, 'flavors': flavors, 'values': values}])

    wsample_full = \
        2*np.random.randint(freq_max, size=nfreqs)+1, \
        2*np.random.randint(freq_max, size=nfreqs)+1, \
        2*np.random.randint(freq_max, size=nfreqs)+1, \
        2*np.random.randint(freq_max, size=nfreqs)+1

    import time
    t1 = time.time()
    res_ref = ft_four_point_obj_ref(worm_config_record, wsample_full, nflavors, beta)
    t2 = time.time()
    res = ft_four_point_obj(worm_config_record, wsample_full, nflavors, beta)
    t3 = time.time()
    print(t2-t1, t3-t2)
    
    np.testing.assert_allclose(res, res_ref)