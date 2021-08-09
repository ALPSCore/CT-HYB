import numpy as np
from numpy.polynomial.legendre import legval
import scipy
import h5py
import os
import irbasis
from irbasis_x.twopoint import FiniteTemperatureBasis, TruncatedBasis
from irbasis_x.freq import check_fermionic

from alpscthyb.util import float_to_complex_array


def load_irbasis(stat, Lambda, beta, cutoff):
    return FiniteTemperatureBasis(
        TruncatedBasis(irbasis.load(stat, Lambda), cutoff=cutoff),
        beta)


def read_param(h5, name):
    if '/parameters/dictionary/'+name in h5:
        return h5['/parameters/dictionary/'+name][()]
    elif '/parameters/'+name in h5:
        return h5['/parameters/'+name][()]
    else:
        raise RuntimeError("Parameter "+ name + " not found") 

#def compute_Tnl(n_matsubara, n_legendre):
    #"""
    #Compute transformation matrix from Legendre to fermionic/bosonic Matsubara frequency
    #Implement Eq. (4.5) in the Boehnke's  Ph.D thesis
    #"""
    #Tnl = np.zeros((n_matsubara, n_legendre), dtype=complex)
    #for n in range(n_matsubara):
        #sph_jn = np.array([scipy.special.spherical_jn(l, (n+0.5)*np.pi) for l in range(n_legendre)])
        #for il in range(n_legendre):
            #Tnl[n,il] = ((-1)**n) * ((1J)**(il+1)) * np.sqrt(2*il + 1.0) * sph_jn[il]
    #return Tnl

def legendre_to_tau(gl, tau, beta):
    """
    Evaluate fermionic/bosonic correaltion function in Legendre basis on tau
    The first axis of gl must correspond to the Legendre basis
    """
    assert all(tau >= 0) and all(tau <= beta)
    nl = gl.shape[0]
    coeff = np.sqrt(2*np.arange(nl)+1.)/beta
    x = 2 * tau/beta - 1
    
    if gl.ndim == 1:
        return legval(x, gl * coeff)
    else:
        rest_dim = gl.shape[1:]
        gl = gl.reshape((nl, -1))
        gtau = np.moveaxis(legval(x, coeff[:,None]*gl, tensor=True), -1, 0)
        return gtau.reshape((gtau.shape[0],) + rest_dim)

def compute_Tnl_sparse(vsample, n_legendre):
    """
    Compute transformation matrix from Legendre to fermionic/bosonic Matsubara frequency
    Implement Eq. (4.5) in the Boehnke's  Ph.D thesis
    """
    Tnl = np.zeros((vsample.size, n_legendre), dtype=np.complex128)
    for idx_n, v in enumerate(vsample):
        abs_v = abs(v)
        sph_jn = np.array(
            [scipy.special.spherical_jn(l, 0.5*abs_v*np.pi) for l in range(n_legendre)])
        for il in range(n_legendre):
            Tnl[idx_n, il] = (1J**(abs_v+il)) * np.sqrt(2*il + 1.0) * sph_jn[il]
        if v < 0:
            Tnl[idx_n, :] = Tnl[idx_n, :].conj()
    return Tnl

def legendre_to_matsubara(gl, vsample):
    Tnl = compute_Tnl_sparse(vsample, gl.shape[0])
    return np.einsum('wl,l...->w...', Tnl, gl)


def load_cmplx(h5, path):
    return float_to_complex_array(h5[path][()])

def _fit_iw(basis_ir, giw):
    if giw.ndim == 1:
        return basis_ir.fit_iw(giw)
    else:
        rest_dim = giw.shape[1:]
        giw = giw.reshape((giw.shape[0],-1))
        gl = basis_ir.fit_iw(giw)
        return gl.reshape((gl.shape[0],) + rest_dim)

def exits_mc_result(h5, name):
    return '/simulation/results/'+name in h5

def read_mc_result(h5, name):
    return {'mean': h5['/simulation/results/'+name+'/mean/value'][()]}

def read_cmplx_mc_result(h5, name):
    re = h5['/simulation/results/'+name+'_Re/mean/value'][()]
    im = h5['/simulation/results/'+name+'_Im/mean/value'][()]
    return {'mean': re + 1J*im}

def postprocess_G1(h5, verbose=False, **kwargs):
    nflavors = kwargs['nflavors']
    beta = kwargs['beta']

    results = {}
    sign = read_mc_result(h5, 'Sign')['mean']
    w_vol = read_mc_result(h5, 'worm_space_volume_G1')['mean']
    z_vol = read_mc_result(h5, 'Z_function_space_volume')['mean']

    # Legendre
    if verbose:
        print("Reading gl...")
    results['gl'] = (w_vol/(sign * z_vol)) * read_cmplx_mc_result(h5, 'G1')['mean'].\
        reshape((nflavors,nflavors,-1)).transpose((2,0,1))

    # Equal time
    if exits_mc_result(h5, "Equal_time_G1_Re"):
        if verbose:
            print("Reading equal_time_G1...")
        results['equal_time_G1'] =  (w_vol/(sign * z_vol * beta)) * \
            read_cmplx_mc_result(h5, 'Equal_time_G1')['mean'].reshape((nflavors,nflavors))

    # vartheta
    if exits_mc_result(h5, "vartheta_legendre_Re"):
        if verbose:
            print("Reading vartheta_legendre...")
        results['vartheta_legendre'] =  -(w_vol/(beta * sign * z_vol)) * \
            read_cmplx_mc_result(h5, 'vartheta_legendre')['mean'].reshape((-1,nflavors,nflavors))

    if exits_mc_result(h5, "vartheta_Re"):
        if verbose:
            print("Reading vartheta...")
        results['vartheta'] =  -(w_vol/(beta * sign * z_vol)) * \
            read_cmplx_mc_result(h5, 'vartheta')['mean'].reshape((-1,nflavors,nflavors))
        results['vartheta_smpl_freqs'] = h5['vartheta_smpl_freqs'][()]
    
    return results

def postprocess_equal_time_G1(h5, verbose=False, **kwargs):
    nflavors = kwargs['nflavors']
    beta = kwargs['beta']

    results = {}
    sign = read_mc_result(h5, 'Sign')['mean']
    w_vol = read_mc_result(h5, 'worm_space_volume_Equal_time_G1')['mean']
    z_vol = read_mc_result(h5, 'Z_function_space_volume')['mean']

    # Equal time
    if exits_mc_result(h5, "Equal_time_G1_Re"):
        if verbose:
            print("Reading equal_time_G1...")
        results['equal_time_G1'] =  (w_vol/(sign * z_vol * beta)) * \
            read_cmplx_mc_result(h5, 'Equal_time_G1')['mean'].reshape((nflavors,nflavors))

    return results

def postprocess_two_point_ph(h5, verbose=False, **kwargs):
    nflavors = kwargs['nflavors']
    beta = kwargs['beta']
    sign = read_mc_result(h5, 'Sign')['mean']
    w_vol = read_mc_result(h5, 'worm_space_volume_Two_point_PH')['mean']
    z_vol = read_mc_result(h5, 'Z_function_space_volume')['mean']
    return {
        'lambda_legendre':
        (w_vol/(sign * z_vol * beta)) * \
            read_cmplx_mc_result(h5, 'lambda_legendre')['mean'].\
            reshape((-1,nflavors,nflavors,nflavors,nflavors))
    }

def postprocess_two_point_pp(h5, verbose=False, **kwargs):
    nflavors = kwargs['nflavors']
    beta = kwargs['beta']
    sign = read_mc_result(h5, 'Sign')['mean']
    sign = read_mc_result(h5, 'Sign')['mean']
    w_vol = read_mc_result(h5, 'worm_space_volume_Two_point_PP')['mean']
    z_vol = read_mc_result(h5, 'Z_function_space_volume')['mean']
    return {
        'varphi_legendre':
        (w_vol/(sign * z_vol * beta)) * \
            read_cmplx_mc_result(h5, 'varphi_legendre')['mean'].\
            reshape((-1,nflavors,nflavors,nflavors,nflavors))
    }


all_worm_space_names = ['G1', 'G2', 'Equal_time_G1', 'Two_point_PH', 'Two_point_PP']

postprocessors = {
    'G1' : postprocess_G1,
    'Equal_time_G1' : postprocess_equal_time_G1,
    'Two_point_PH' : postprocess_two_point_ph,
    'Two_point_PP' : postprocess_two_point_pp,
}

class QMCResult:
    def __init__(self, p, verbose=False, Lambda=1E+5, cutoff=1e-8) -> None:
        if verbose:
            print(p+'.out.h5')
    
        with h5py.File(p+'.out.h5','r') as h5:
            self.sites = read_param(h5, 'model.sites')
            self.beta = read_param(h5, 'model.beta')
            self.nflavors = 2*self.sites

        with h5py.File(p+'_wormspace_G1.out.h5','r') as h5:
            self.hopping = load_cmplx(h5, 'hopping')
            self.Delta_tau = load_cmplx(h5, '/Delta_tau').transpose((2,0,1))
            # U_tensor: (1/2) U_{ijkl} d^dagger_i d^dagger_j d_k d_l
            self.U_tensor = load_cmplx(h5, 'U_tensor')
            # asymU: (1/4) U_{ikjl} d^dagger_i d^dagger_j d_l d_k
            asymU = 2.0 * self.U_tensor.copy().transpose(0, 3, 1, 2)
            asymU = 0.5 * (asymU - asymU.transpose((0, 3, 2, 1)))
            asymU = 0.5 * (asymU - asymU.transpose((2, 1, 0, 3)))
            self.asymU = asymU
        
        # Read Monte Carlo data
        for ws_name, post_ in postprocessors.items():
            fname = p+f'_wormspace_{ws_name}.out.h5'
            if not os.path.exists(fname):
                continue
            with h5py.File(fname, 'r') as h5:
                res_ = post_(h5, verbose, beta=self.beta, nflavors=self.nflavors)
                for k, v in res_.items():
                    self.__setattr__(k, v)

        # Set up IR basis
        self.basis_f = load_irbasis('F', Lambda, self.beta, cutoff)

        # Fit Delta(tau) with IR basis
        taus = np.linspace(0, self.beta, self.Delta_tau.shape[0])
        all_l = np.arange(self.basis_f.dim())
        Utaul = self.basis_f.Ultau(all_l[:,None], taus[None,:]).T
        Delta_l = np.linalg.lstsq(
                Utaul,
                self.Delta_tau.reshape((-1,self.nflavors**2)),
                rcond=None
            )[0]
        self.Delta_l = Delta_l.reshape((-1, self.nflavors, self.nflavors))

    def compute_gir_SIE(self):
        """
        Reconstruct one-particle Green's function using SIE
        in fermionic IR
        """
        vsample = self.basis_f.wsample
        giv = self.compute_giv_SIE(vsample)
        return _fit_iw(self.basis_f, giv)

    def compute_g0iv(self, vsample):
        """
        Compute non-interacting one-particle Green's function
        on Matsubara frequencies
        """
        vsample = check_fermionic(vsample)
        nfreqs = vsample.size
        
        # Compute Delta(iv)
        Delta_iv = self.basis_f.evaluate_iw(self.Delta_l, vsample)

        # Compute non-interacting Green's function
        G0 = np.empty((nfreqs, self.nflavors, self.nflavors), dtype=np.complex128)
        I = np.identity(self.nflavors)
        for ifreq, v in enumerate(vsample):
            iv = 1J * v * np.pi/self.beta
            G0[ifreq, ...] = np.linalg.inv(iv * I - Delta_iv[ifreq, ...] - self.hopping[None,:,:])
        return G0

    def compute_giv_SIE(self, vsample):
        """
        Reconstruct one-particle Green's function using SIE
        on Matsubara frequencies
        """
        vsample = check_fermionic(vsample)
        nfreqs = vsample.size
        
        # Compute vartheta
        vartheta = legendre_to_matsubara(self.vartheta_legendre, vsample)

        # Compute Delta(iv)
        Delta_iv = self.basis_f.evaluate_iw(self.Delta_l, vsample)

        # Compute A_{ab}
        A = self.hopping + np.einsum('abij,ij->ab', self.asymU, self.equal_time_G1)

        # Compute calG = (iv - Delta(iv)), G0, full G, self-energy
        #print("vartheta", vartheta)
        #print("A", A)
        #vartheta[...] = 0.0 #debug
        G = np.empty((nfreqs, self.nflavors, self.nflavors), dtype=np.complex128)
        Sigma = np.empty_like(G)
        I = np.identity(self.nflavors)
        for ifreq, v in enumerate(vsample):
            iv = 1J * v * np.pi/self.beta
            calG = np.linalg.inv(iv * I - Delta_iv[ifreq, ...])
            G[ifreq, ...] = calG + \
                np.einsum('ai,ij,jb->ab', calG,  (A + vartheta[ifreq,...]), calG, optimize=True)
            invG0 = iv * I - Delta_iv[ifreq, ...] - self.hopping
            invG  = np.linalg.inv(G[ifreq, ...])
            Sigma[ifreq, ...] = invG0 - invG
        
        return G 
    
    def compute_sigma_iv(self, giv, vsample):
        vsample = check_fermionic(vsample)
        Delta_iv = self.basis_f.evaluate_iw(self.Delta_l, vsample)
        Sigma = np.empty((vsample.size, self.nflavors, self.nflavors), dtype=np.complex128)
        I = np.identity(self.nflavors)
        for ifreq, v in enumerate(vsample):
            iv = 1J * v * np.pi/self.beta
            invG0 = iv * I - Delta_iv[ifreq, ...] - self.hopping
            invG  = np.linalg.inv(giv[ifreq, ...])
            Sigma[ifreq, ...] = invG0 - invG
        return Sigma
    
    def compute_giv_from_legendre(self, vsample):
        vsample = check_fermionic(vsample)
        Tnl = compute_Tnl_sparse(vsample, self.gl.shape[0])
        giv = np.einsum('nl,lij->nij', Tnl, self.gl, optimize=True)
        return giv
