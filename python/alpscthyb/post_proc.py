from itertools import product
import numpy as np
from numpy.polynomial.legendre import legval
import scipy
import h5py
import os
import irbasis
from irbasis_x.twopoint import FiniteTemperatureBasis, TruncatedBasis
from irbasis_x.freq import check_bosonic, check_fermionic, check_full_convention
from scipy.sparse import coo
from alpscthyb.exact_diag import construct_cdagger_ops, compute_fermionic_2pt_corr_func, \
    compute_3pt_corr_func, construct_ham, compute_bosonic_2pt_corr_func, compute_4pt_corr_func, \
        compute_expval

from alpscthyb.util import float_to_complex_array
from alpscthyb import mpi

def _einsum(subscripts, *operands):
    return np.einsum(subscripts, *operands, optimize=True)

def _check_full_convention(*wsample_full):
    assert all(wsample_full[0] -wsample_full[1] + wsample_full[2] - wsample_full[3] == 0)
    return check_full_convention(*wsample_full)

class WormConfigRecord:
    def __init__(self, dirname, num_time_idx, num_flavor_idx) -> None:
        self.datasets = []
        for file in os.listdir(dirname):
            if not file.endswith(".h5"):
              continue
            with h5py.File(dirname+"/"+file, "r") as h5:
                local_slice = mpi.get_slice(h5[f'/taus/0'].size)
                taus = []
                for t in range(num_time_idx):
                    taus.append(h5[f'/taus/{t}'][local_slice][()])

                flavors = []
                for f in range(num_flavor_idx):
                    flavors.append(h5[f'/flavors/{f}'][local_slice][()])

                values = h5[f'/vals_real'][local_slice][()] \
                    + 1J* h5[f'/vals_imag'][local_slice][()]

                self.datasets.append(
                    {
                        'taus':    taus,
                        'flavors': flavors,
                        'values' : values
                    }
                )

def ft_three_point_obj(worm_config_record, wsample, nflavors, beta):
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

def ft_four_point_obj(worm_config_record, wsample, nflavors, beta):
    wf1, wf2, wf3, wf4 = wsample
    wf1 = check_fermionic(wf1)
    wf2 = check_fermionic(wf2)
    wf3 = check_fermionic(wf3)
    wf4 = check_fermionic(wf4)
    res = np.zeros((wf1.size,) + 4*(nflavors,), dtype=np.complex128)
    ndata = 0
    for dset in worm_config_record.datasets:
        exp_ = np.exp(1J * np.pi * (
                +wf1[:,None] * dset['taus'][0][None,:]
                -wf2[:,None] * dset['taus'][1][None,:]
                +wf3[:,None] * dset['taus'][2][None,:]
                -wf4[:,None] * dset['taus'][3][None,:]
                )/beta)
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
    return res/ndata

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
    if verbose:
        print("Reading varphi_legendre...")
    return {
        'varphi_legendre':
        (w_vol/(sign * z_vol * beta)) * \
            read_cmplx_mc_result(h5, 'varphi_legendre')['mean'].\
            reshape((-1,nflavors,nflavors,nflavors,nflavors))
    }

def postprocess_three_point_ph(h5, verbose=False, **kwargs):
    sign = read_mc_result(h5, 'Sign')['mean']
    w_vol = read_mc_result(h5, 'worm_space_volume_Three_point_PH')['mean']
    z_vol = read_mc_result(h5, 'Z_function_space_volume')['mean']
    if verbose:
        print("Reading eta...")
    return {
        'eta_coeff' : w_vol/(sign * z_vol),
        'eta_datasets' :
            WormConfigRecord(kwargs['prefix'] + "_wormspace_Three_point_PH_eta_results", 3, 4)
    }

def postprocess_three_point_pp(h5, verbose=False, **kwargs):
    sign = read_mc_result(h5, 'Sign')['mean']
    w_vol = read_mc_result(h5, 'worm_space_volume_Three_point_PP')['mean']
    z_vol = read_mc_result(h5, 'Z_function_space_volume')['mean']
    if verbose:
        print("Reading gamma...")
    return {
        'gamma_coeff' : w_vol/(sign * z_vol),
        'gamma_datasets' :
            WormConfigRecord(kwargs['prefix'] + "_wormspace_Three_point_PP_gamma_results", 3, 4)
    }

def postprocess_G2(h5, verbose=False, **kwargs):
    beta = kwargs['beta']
    prefix = kwargs['prefix']
    sign = read_mc_result(h5, 'Sign')['mean']
    w_vol = read_mc_result(h5, 'worm_space_volume_G2')['mean']
    z_vol = read_mc_result(h5, 'Z_function_space_volume')['mean']
    return {
        'h_corr_coeff': w_vol/(sign * z_vol),
        'h_corr_datasets':
            WormConfigRecord(prefix + "_wormspace_G2_h_corr_results", 4, 4)
    }


postprocessors = {
    'G1'             : postprocess_G1,
    'G2'             : postprocess_G2,
    'Equal_time_G1'  : postprocess_equal_time_G1,
    'Two_point_PH'   : postprocess_two_point_ph,
    'Two_point_PP'   : postprocess_two_point_pp,
    'Three_point_PH' : postprocess_three_point_ph,
    'Three_point_PP' : postprocess_three_point_pp,
}

class VertexEvaluator(object):
    def __init__(self):
        pass

    def get_asymU(self):
        return NotImplementedError

    def compute_giv(self, wfs):
        return NotImplementedError

    def get_dm(self):
        """ One-particle density matrix """
        return NotImplementedError

    def compute_vartheta(self, wfs):
        raise NotImplementedError

    def compute_lambda(self, wbs):
        raise NotImplementedError

    def compute_varphi(self, wbs):
        raise NotImplementedError
    
    def compute_eta(self, wfs, wbs):
        raise NotImplementedError
    
    def compute_gamma(self, wfs, wbs):
        raise NotImplementedError

    def compute_h(self, wsample_full):
        raise NotImplementedError

    def compute_v(self):
        return self.hopping + _einsum('abij,ij->ab', self.get_asymU(), self.get_dm())

    def compute_calgiv(self, wfs):
        return self._compute_non_int_giv(wfs, np.zeros((self.nflavors, self.nflavors)))

    def _compute_non_int_giv(self, wfs, hopping):
        """ Compute non-interacting Green's function for given hopping matrix"""
        wfs = check_fermionic(wfs)
        nfreqs = wfs.size
        
        # Compute Delta(iv)
        Delta_iv = self.basis_f.evaluate_iw(self.Delta_l, wfs)

        # Compute non-interacting Green's function
        G0 = np.empty((nfreqs, self.nflavors, self.nflavors), dtype=np.complex128)
        I = np.identity(self.nflavors)
        for ifreq, v in enumerate(wfs):
            iv = 1J * v * np.pi/self.beta
            G0[ifreq, ...] = np.linalg.inv(iv * I - Delta_iv[ifreq, ...] - hopping[None,:,:])
        return G0

    def compute_xi(self, wfs):
        """ Compute xi(iv) """
        wfs = check_fermionic(wfs)
        return _einsum(
            'wai,wib->wab',
            (self.compute_v()[None,:,:] + self.compute_vartheta(wfs)),
            self.compute_calgiv(wfs)
        )
    
    def compute_phi(self, wbs):
        wbs = check_bosonic(wbs)
        phi = 0.25 * \
            _einsum(
                'abij,cdkl,wijkl->wabcd',
                self.get_asymU(),
                self.get_asymU(),
                self.compute_lambda(wbs)
            )
        const = 0.25 * self.beta * (
            np.einsum('ab,cd->abcd', self.hopping, self.hopping) +
            np.einsum('ab,cdkl,kl->abcd', self.hopping, self.get_asymU(), self.get_dm(), optimize=True) +
            np.einsum('cd,abij,ij->abcd', self.hopping, self.get_asymU(), self.get_dm(), optimize=True)
        )
        phi[wbs==0] += const[None,:,:,:,:]
        return phi

    def compute_Psi(self, wbs):
        wbs = check_bosonic(wbs)
        return 0.25 * \
            _einsum('ajci,kbld,Wijkl->Wabcd',
                self.get_asymU(), self.get_asymU(),
                self.compute_varphi(wbs))

    def compute_f(self, wfs, wbs):
        wfs = check_fermionic(wfs)
        wbs = check_bosonic(wbs)
        f = 0.5 * np.einsum(
            'Wabij,cdij->Wabcd',
            self.compute_eta(wfs, wbs),
            self.get_asymU(),
            optimize=True
        )
        f += -0.5 * self.beta * \
            _einsum('W,Wab,cd->Wabcd',
                wbs == 0,
                self.compute_vartheta(wfs),
                self.hopping
            )
        return f
    
    def compute_g(self, wfs, wfs_p):
        wfs = check_fermionic(wfs)
        wfs_p = check_fermionic(wfs_p)
        return 0.5 * _einsum('jbkd,Wacjk->Wabcd',
           self.get_asymU(),
           self.compute_gamma(wfs, wfs + wfs_p)
        )
    
    def compute_F(self, wsample_full, verbose=False):
        wsample_full = _check_full_convention(*wsample_full)
        v1, v2, v3, v4 = wsample_full
        beta = self.beta
        asymU = self.get_asymU()
        scrF = np.zeros((wsample_full[0].size,) + 4*(self.nflavors,), dtype=np.complex128)

        vab = self.compute_v()

        scrF += beta * asymU[None, ...]

        v1_ = self.compute_vartheta(v1) + vab[None,:,:]
        v3_ = self.compute_vartheta(v3) + vab[None,:,:]
        scrF += (beta**2) * np.einsum('W,Wab,Wcd->Wabcd', v1==v2, v1_, v3_, optimize=True)
        scrF -= (beta**2) * np.einsum('W,Wad,Wcb->Wabcd', v1==v4, v1_, v3_, optimize=True)

        # xi
        scrF += beta * _einsum('ibcd,Wai->Wabcd', asymU, self.compute_xi(v1))
        scrF += beta * _einsum('aicd,Wbi->Wabcd', asymU, self.compute_xi(-v2).conj())
        scrF += beta * _einsum('abid,Wci->Wabcd', asymU, self.compute_xi(v3))
        scrF += beta * _einsum('abci,Wdi->Wabcd', asymU, self.compute_xi(-v4).conj())

        # phi
        scrF += -4 * beta * self.compute_phi(v1-v2)
        scrF +=  4 * beta * _einsum('Wadcb->Wabcd', self.compute_phi(v1-v4))

        # f
        scrF +=  2 * beta * self.compute_f(v1, v1-v2)
        scrF +=  2 * beta * _einsum('Wcdab->Wabcd', self.compute_f(v3, v2-v1))
        scrF += -2 * beta * _einsum('Wadcb->Wabcd', self.compute_f(v1, v1-v4))
        scrF += -2 * beta * _einsum('Wcbad->Wabcd', self.compute_f(v3, v4-v1))

        # g
        scrF += -beta * self.compute_g(v1, v3)
        scrF += -beta * _einsum('Wdcba->Wabcd', self.compute_g(-v4, -v2).conj())

        # Psi
        scrF += -beta * self.compute_Psi(v1+v3)

        # h
        scrF -= self.compute_h(wsample_full)

        # Replace legs from calG to G
        r1 = self._invG_calG(v1)
        r2 = self._calG_invG(v2)
        r3 = self._invG_calG(v3)
        r4 = self._calG_invG(v4)
        return _einsum('waA,wBb,wcC,wDd,wABCD->wabcd', r1, r2, r3, r4, scrF)

    def _invG_calG(self, wfs):
        wfs_unique, wfs_where = np.unique(wfs, return_inverse=True)
        res_unique = np.empty((wfs_unique.size, self.nflavors, self.nflavors), dtype=np.complex128)
        giv = self.compute_giv(wfs_unique)
        calgiv = self.compute_calgiv(wfs_unique)
        for i in range(wfs_unique.size):
            res_unique[i,:,:] = np.linalg.inv(giv[i,...]) @ calgiv[i,...]
        return res_unique[wfs_where, ...]

    def _calG_invG(self, wfs):
        wfs_unique, wfs_where = np.unique(wfs, return_inverse=True)
        res_unique = np.empty((wfs_unique.size, self.nflavors, self.nflavors), dtype=np.complex128)
        giv = self.compute_giv(wfs_unique)
        calgiv = self.compute_calgiv(wfs_unique)
        for i in range(wfs_unique.size):
            res_unique[i,:,:] = calgiv[i,...] @ np.linalg.inv(giv[i,...])
        return res_unique[wfs_where, ...]


class VertexEvaluatorU0(VertexEvaluator):
    """
    Non-interacting limit
    """
    def __init__(self, nflavors, beta, basis_f, basis_b, hopping, Delta_l, asymU=None):
        super().__init__()
        self.nflavors = nflavors
        self.beta = beta
        self.basis_f = basis_f
        self.basis_b = basis_b
        self.Delta_l = Delta_l
        self.hopping = hopping
        if asymU is None:
            self.asymU = np.zeros(4*(self.nflavors,))
        else:
            self.asymU = asymU

        self.dm = -self.compute_gtau([self.beta]).reshape((nflavors,nflavors)).T

        assert (self.hopping == self.hopping.T.conj()).all()

    def get_asymU(self):
        return self.asymU

    def get_dm(self):
        return self.dm

    def compute_gtau(self, taus):
        giv = self.compute_giv(self.basis_f.wsample)
        gl = _fit_iw(self.basis_f, giv)
        Ftau = self.basis_f.Ultau_all_l(taus).T
        return _einsum('tl,lab->tab', Ftau, gl)

    def compute_giv(self, wfs):
        return self._compute_non_int_giv(wfs, self.hopping)

    def compute_vartheta(self, wfs):
        """ Compute vartheta(wfs) """
        wfs = check_fermionic(wfs)
        return np.einsum(
            'ai,jb,wij->wab',
            self.hopping,
            self.hopping, self.compute_giv(wfs))

    def compute_lambda(self, wbs):
        """ Compute lambda(wbs) """
        wbs = check_bosonic(wbs)

        taus = self.basis_b.sampling_points_tau(self.basis_b.dim()-1)
        Ftau = self.basis_b.Ultau_all_l(taus).T
        lambda_tau = _einsum(
            'Tda,Tbc->Tabcd',
            self.compute_gtau(self.beta - taus),
            self.compute_gtau(taus)
        )

        lambda_l = _einsum('lt,t...->l...', np.linalg.pinv(Ftau), lambda_tau)
        lambda_wb = self.basis_b.evaluate_iw(lambda_l, wsample=wbs)
        lambda_wb[wbs==0,...] += self.beta * _einsum('ab,cd->abcd', self.get_dm(), self.get_dm())[None,...]
        return lambda_wb


    def compute_varphi(self, wbs):
        taus = self.basis_b.sampling_points_tau(self.basis_b.dim()-1)
        Ftau = self.basis_b.Ultau_all_l(taus).T
        gtau = self.compute_gtau(taus)
        varphi_tau = \
            _einsum('Tad,Tbc->Tabcd', gtau, gtau) - \
            _einsum('Tac,Tbd->Tabcd', gtau, gtau)
        varphi_l = _einsum('lt,t...->l...', np.linalg.pinv(Ftau), varphi_tau)
        return self.basis_b.evaluate_iw(varphi_l, wsample=wbs)


    def compute_eta(self, wfs, wbs):
        """ Compute eta(wfs, wbs) """
        wfs = check_fermionic(wfs)
        wbs = check_bosonic(wbs)
        eta1 = - self.beta * \
           np.einsum('w,wab,cd->wabcd',
               (wbs == 0),
               self.compute_vartheta(wfs),
               self.dm
           )
        eta2 = np.einsum('wac,wbd->wabcd',
           self.compute_xi(wfs),
           self.compute_xi(wbs-wfs).conj()
        )
        return eta1 + eta2
    
    def compute_gamma(self, wfs, wbs):
        """ Compute gamma(wfs, wbs) """
        wfs = check_fermionic(wfs)
        wbs = check_bosonic(wbs)
        xi1 = self.compute_xi(wfs)
        xi2 = self.compute_xi(wbs-wfs)
        return np.einsum('wad,wbc->wabcd', xi1, xi2) - np.einsum('wac,wbd->wabcd', xi1, xi2)

    def compute_h(self, wsample_full):
        """ Compute h(v1, v2, v3, v4) """
        v1, v2, v3, v4 = _check_full_convention(*wsample_full)
        vartheta1 = self.compute_vartheta(v1)
        vartheta3 = self.compute_vartheta(v3)
        return (self.beta**2) * (
            np.einsum('w,wab,wcd->wabcd', (v1==v2), vartheta1, vartheta3) -
            np.einsum('w,wad,wcb->wabcd', (v1==v4), vartheta1, vartheta3)
        )

class VertexEvaluatorAtomED(VertexEvaluator):
    """
    Exact diagonalization for atom
    """
    def __init__(self, nflavors, beta, hopping, asymU):
        super().__init__()
        self.nflavors = nflavors
        self.beta = beta
        self.hopping = hopping
        self.asymU = asymU

        _, self.cdag_ops = construct_cdagger_ops(nflavors)
        self.c_ops = [op.transpose(copy=True) for op in self.cdag_ops]
        self.ham = construct_ham(hopping, self.asymU, self.cdag_ops)
        self.evals, self.evecs = np.linalg.eigh(self.ham.toarray())
        self.q_ops = [c@self.ham-self.ham@c for c in self.c_ops]
        self.qdag_ops = [self.ham@cdag-cdag@self.ham for cdag in self.cdag_ops]

        self.dm = np.zeros((nflavors,nflavors), dtype=np.complex128)
        for i, j in product(range(nflavors), repeat=2):
            self.dm[i,j] = compute_expval(
                self.cdag_ops[i]@self.c_ops[j], beta, self.evals, self.evecs)

        self.vab = np.empty((self.nflavors, self.nflavors), dtype=object)
        for a, b in product(range(self.nflavors), repeat=2):
            self.vab[a,b] = hopping[a,b] * \
                    scipy.sparse.identity(2**self.nflavors, dtype=np.complex128, format='coo')
            for i, j in product(range(self.nflavors), repeat=2):
                self.vab[a,b]+= self.asymU[a,b,i,j] * (self.cdag_ops[i]@self.c_ops[j])

    def get_asymU(self):
        return self.asymU

    def get_dm(self):
        return self.dm

    def compute_giv(self, wfs):
        wfs = check_fermionic(wfs)
        giv = np.zeros((wfs.size,self.nflavors, self.nflavors), dtype=np.complex128)
        for i, j in product(range(self.nflavors), repeat=2):
            giv[:,i,j] = compute_fermionic_2pt_corr_func(
                self.c_ops[i], self.cdag_ops[j], self.beta, wfs, self.evals, self.evecs)
        return giv

    def compute_vartheta(self, wfs):
        """ Compute vartheta(wfs) """
        wfs = check_fermionic(wfs)
        vartheta = np.zeros((wfs.size, self.nflavors, self.nflavors), dtype=np.complex128)
        for i, j in product(range(self.nflavors), repeat=2):
            vartheta[:,i,j] = compute_fermionic_2pt_corr_func(
                self.q_ops[i], self.qdag_ops[j], self.beta, wfs, self.evals, self.evecs)
        return vartheta

    def compute_phi(self, wbs):
        wbs = check_bosonic(wbs)
        phi = np.zeros((wbs.size,)+ 4*(self.nflavors,), dtype=np.complex128)
        for i, j, k, l in product(range(self.nflavors),repeat=4):
            phi[:,i,j,k,l] = \
                0.25 * compute_bosonic_2pt_corr_func(
                    self.vab[i,j], self.vab[k,l], self.beta, wbs, self.evals, self.evecs)
        return phi                

    """
    def compute_Psi(self, wbs):
        wbs = check_bosonic(wbs)
        tmp = np.zeros((wbs.size,)+ 4*(self.nflavors,), dtype=np.complex128)
        #print("wbs", wbs)
        for i, j, k, l in product(range(self.nflavors),repeat=4):
            tmp[:,i,j,k,l] = \
                compute_bosonic_2pt_corr_func(
                    self.c_ops[i]@self.c_ops[j],
                    self.cdag_ops[k]@self.cdag_ops[l],
                    self.beta, wbs, self.evals, self.evecs)
            #print(i, j, k, l, tmp[:,i,j,k,l])

        #print()
        #print("tmp", tmp)
        #print(-0.25 * _einsum('ajck,JbKd,WkjJK->Wabcd', self.get_asymU(), self.get_asymU(), tmp))
        return -0.25 * _einsum('ajck,JbKd,WkjJK->Wabcd', self.get_asymU(), self.get_asymU(), tmp)                
        #return 0.0 * _einsum('ajck,JbKd,WkjJK->Wabcd', self.get_asymU(), self.get_asymU(), tmp)                
    """

    def compute_lambda(self, wbs):
        """ Compute lambda(wbs) """
        wbs = check_bosonic(wbs)
        lambda_wb = np.zeros((wbs.size,)+ 4*(self.nflavors,), dtype=np.complex128)
        for i, j, k, l in product(range(self.nflavors),repeat=4):
            lambda_wb[:,i,j,k,l] = \
                compute_bosonic_2pt_corr_func(
                    self.cdag_ops[i]@self.c_ops[j],
                    self.cdag_ops[k]@self.c_ops[l],
                    self.beta, wbs, self.evals, self.evecs)
        return lambda_wb                

    def compute_varphi(self, wbs):
        """ Compute varphi(wbs) """
        wbs = check_bosonic(wbs)
        varphi_wb = np.zeros((wbs.size,) + 4*(self.nflavors,), dtype=np.complex128)
        for i, j, k, l in product(range(self.nflavors),repeat=4):
            varphi_wb[:,i,j,k,l] = \
                compute_bosonic_2pt_corr_func(
                    self.c_ops[i]@self.c_ops[j],
                    self.cdag_ops[k]@self.cdag_ops[l],
                    self.beta, wbs, self.evals, self.evecs)
        return varphi_wb                

    def compute_eta(self, wfs, wbs):
        """ Compute eta(wfs, wbs) """
        wfs = check_fermionic(wfs)
        wbs = check_bosonic(wbs)
        eta = np.zeros((wfs.size,)+(self.nflavors,)*4, dtype=np.complex128)
        for i,j,k,l in product(range(self.nflavors), repeat=4):
            eta[:,i,j,k,l] = compute_3pt_corr_func(
                self.q_ops[i], self.qdag_ops[j], self.cdag_ops[k]@self.c_ops[l],
                self.beta, (wfs,wbs), self.evals, self.evecs)
        return eta

    def compute_gamma(self, wfs, wbs):
        """ Compute eta(wfs, wbs) """
        wfs = check_fermionic(wfs)
        wbs = check_bosonic(wbs)
        gamma = np.zeros((wfs.size,)+(self.nflavors,)*4, dtype=np.complex128)
        for i,j,k,l in product(range(self.nflavors), repeat=4):
            gamma[:,i,j,k,l] = compute_3pt_corr_func(
                self.q_ops[i], self.q_ops[j], self.cdag_ops[k]@self.cdag_ops[l],
                self.beta, (wfs,wbs), self.evals, self.evecs)
        return gamma
    
    def compute_h(self, wsample_full):
        wsample_full = _check_full_convention(*wsample_full)
        h = np.empty((wsample_full[0].size,) + 4*(self.nflavors,), dtype=np.complex128)
        for i, j, k, l in product(range(self.nflavors),repeat=4):
            h[:,i,j,k,l] = compute_4pt_corr_func(
                self.q_ops[i], self.qdag_ops[j], self.q_ops[k], self.qdag_ops[l],
                self.beta, wsample_full, self.evals, self.evecs
            )
        return h
    
    def compute_g4pt(self, wsample_full):
        wsample_full = _check_full_convention(*wsample_full)
        g4pt = np.empty((wsample_full[0].size,) + 4*(self.nflavors,), dtype=np.complex128)
        for i, j, k, l in product(range(self.nflavors),repeat=4):
            g4pt[:,i,j,k,l] = compute_4pt_corr_func(
                self.c_ops[i], self.cdag_ops[j], self.c_ops[k], self.cdag_ops[l],
                self.beta, wsample_full, self.evals, self.evecs
            )
        return g4pt


    def _compute_non_int_giv(self, wfs, hopping):
        # Compute non-interacting Green's function
        wfs = check_fermionic(wfs)
        nfreqs = wfs.size
        G0 = np.empty((nfreqs, self.nflavors, self.nflavors), dtype=np.complex128)
        I = np.identity(self.nflavors)
        for ifreq, v in enumerate(wfs):
            iv = 1J * v * np.pi/self.beta
            G0[ifreq, ...] = np.linalg.inv(iv * I - hopping[None,:,:])
        return G0

class QMCResult(VertexEvaluator):
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
                res_ = post_(h5, verbose, beta=self.beta, nflavors=self.nflavors, prefix=p)
                for k, v in res_.items():
                    self.__setattr__(k, v)

        # Set up IR basis
        self.basis_f = load_irbasis('F', Lambda, self.beta, cutoff)
        self.basis_b = load_irbasis('B', Lambda, self.beta, cutoff)

        # Fit Delta(tau) with IR basis
        taus = np.linspace(0, self.beta, self.Delta_tau.shape[0])
        all_l = np.arange(self.basis_f.dim())
        Ftau = self.basis_f.Ultau(all_l[:,None], taus[None,:]).T
        regularizer = self.basis_f.Sl(all_l)
        Delta_l = np.linalg.lstsq(
                Ftau * regularizer[None,:],
                self.Delta_tau.reshape((-1,self.nflavors**2)),
                rcond=None
            )[0] * regularizer[:,None]
        self.Delta_l = Delta_l.reshape((-1, self.nflavors, self.nflavors))
        self.Delta_tau_rec = np.einsum('tl,lij->tij', Ftau, self.Delta_l)

    def get_asymU(self):
        return self.asymU

    def get_dm(self):
        return self.equal_time_G1

    def compute_giv(self, wfs):
        return self.compute_giv_SIE(wfs)

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


    def compute_eta(self, wfs, wbs):
        wfs = check_fermionic(wfs)
        wbs = check_bosonic(wbs)
        return self.eta_coeff * \
            ft_three_point_obj(self.eta_datasets, (wfs,wbs), self.nflavors, self.beta)
    
    def compute_gamma(self, wfs, wbs):
        wfs = check_fermionic(wfs)
        wbs = check_bosonic(wbs)
        return self.gamma_coeff * \
            ft_three_point_obj(self.gamma_datasets, (wfs,wbs), self.nflavors, self.beta)

    def compute_h(self, wsample_full):
        wsample_full = _check_full_convention(*wsample_full)
        return self.h_corr_coeff * \
            ft_four_point_obj(self.h_corr_datasets, wsample_full, self.nflavors, self.beta)

    def compute_lambda(self, wbs):
        return legendre_to_matsubara(self.lambda_legendre, wbs)

    def compute_varphi(self, wbs):
        return legendre_to_matsubara(self.varphi_legendre, wbs)

    def compute_vartheta(self, wfs):
        """ Compute vartheta(wfs) """
        wfs = check_fermionic(wfs)
        return legendre_to_matsubara(self.vartheta_legendre, wfs)
