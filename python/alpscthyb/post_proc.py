from itertools import product
import numpy as np
from numpy.polynomial.legendre import legval
import scipy
import h5py
import os
import irbasis
import irbasis_x
from irbasis_x.twopoint import FiniteTemperatureBasis, TruncatedBasis
from irbasis_x.freq import check_bosonic, check_fermionic, check_full_convention
from alpscthyb.exact_diag import compute_fermionic_2pt_corr_func, \
    compute_3pt_corr_func, construct_ham, compute_bosonic_2pt_corr_func, compute_4pt_corr_func, \
        compute_expval
from alpscthyb.occupation_basis import construct_cdagger_ops

from alpscthyb.util import float_to_complex_array
from alpscthyb.interaction import mk_asymm
from alpscthyb import mpi

import time

def _einsum(subscripts, *operands):
    return np.einsum(subscripts, *operands, optimize=True)

def _check_full_convention(*wsample_full):
    assert all(wsample_full[0] -wsample_full[1] + wsample_full[2] - wsample_full[3] == 0)
    return check_full_convention(*wsample_full)


def _stable_fit(A, data, rcond=1e-10):
    U, s, Vt = np.linalg.svd(A)
    dim = np.sum(s/s[0] > rcond)
    U = U[:,0:dim]
    s = s[0:dim]
    Vt = Vt[0:dim,:]

    tmp1 = np.einsum('ij,j...->i...', U.conj().T, data)
    tmp2 = np.einsum('i,i...->i...', 1/s, tmp1)
    return np.einsum('ij,j...->i...', Vt.conj().T, tmp2)


class WormConfigRecord:
    def __init__(self, datasets):
        self.datasets = datasets

    @classmethod
    def load(cls, dirname, num_time_idx, num_flavor_idx):
        datasets = []
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

                datasets.append(
                    {
                        'taus':    taus,
                        'flavors': flavors,
                        'values' : values
                    }
                )
        return WormConfigRecord(datasets)
    

def _ft_impl2(dset, wsample, beta, phase_calc):
    """ Naive Fourier transformation without normalization """
    phase = phase_calc(beta, wsample, dset['taus'])
    res_ = dset['values'][None,:] * phase
    return np.sum(res_, axis=1)

def _subset_dset(dset, idx):
    """ Exact subset of dset """
    dset_ = {}
    dset_['taus'] = [taus_[idx] for taus_ in dset['taus']]
    dset_['values'] = dset['values'][idx]
    return dset_

def _ft_impl(dset, wsample, beta, phase_calc, max_work_mem=1e+9):
    """ 
    Naive Fourier transformation for a spefic flavor combination
    without normalization

    max_worm_mem: float
       Max work memory in MB
    """
    nw = wsample[0].size
    nmax_config = int(max_work_mem/nw/16/3)
    nconfig = dset['values'].size
    nsplit = int(np.ceil(nconfig / nmax_config))
    res = np.zeros(nw, dtype=np.complex128)
    for idx_ in np.array_split(np.arange(nconfig), nsplit):
        dset_ = _subset_dset(dset, idx_)
        res += _ft_impl2(dset_, wsample, beta, phase_calc)
    return res

def _ft(worm_config_record, wsample, nflavors, beta, phase_calc, max_work_mem=1E+9):
    """ Fourier transform of three-point/four-point objects """
    nw = wsample[0].size
    res = np.zeros((nw,) + 4*(nflavors,), dtype=np.complex128)
    ndata = 0
    for dset in worm_config_record.datasets:
        ndata += dset['values'].size
        for f1, f2, f3, f4 in product(range(nflavors), repeat=4):
            flavors_data = dset['flavors']
            where = \
                np.logical_and(
                    np.logical_and(flavors_data[0] == f1, flavors_data[1] == f2),
                    np.logical_and(flavors_data[2] == f3, flavors_data[3] == f4)
                )
            idx_ = np.where(where)[0]
            if idx_.size == 0:
                continue
            dset_ = _subset_dset(dset, idx_)
            res[:,f1,f2,f3,f4] += _ft_impl(dset_, wsample, beta, phase_calc, max_work_mem)

    res = mpi.allreduce(res)
    ndata = mpi.allreduce(ndata)
    return res/ndata

def _ft_unique_freqs(worm_config_record, wsample, nflavors, beta, phase_calc, max_work_mem=1E+9):
    """
    Fourier transform of three-point/four-point objects
    Compute only for unique frequency combinations
    """
    w = irbasis_x._aux.collect(*wsample)
    w_unique, w_where = np.unique(w, return_inverse=True)
    #print("debug", w.size, w_unique.size)
    res_unique = _ft(
        worm_config_record,
        irbasis_x._aux.split(w_unique, len(wsample)),
        nflavors, beta, phase_calc, max_work_mem)
    return res_unique[w_where, ...]

def _eval_exp(beta, wf, taus, sign):
    """ Exvaluate exp(1J*sign*PI*wf*taus/beta)"""
    wf_unique, wf_where = np.unique(wf, return_inverse=True)
    coeff = sign * 1J * np.pi/beta
    exp_unique = np.exp(coeff * wf_unique[:,None] * taus[None,:])
    return exp_unique[wf_where, :]

def _phase_calc_three_point(beta, wsample, taus):
    """ Compute phase for Fourier transform of three-point obj in ph channel """
    wfs, wbs = wsample
    wfs = check_fermionic(wfs)
    wbs = check_bosonic(wbs)
    taus_f = taus[0] - taus[1]
    taus_b = taus[1] - taus[2]
    exp_ = np.ones((wfs.size, taus_f.size), dtype=np.complex128)
    exp_ *= _eval_exp(beta, wfs, taus_f,  1)
    exp_ *= _eval_exp(beta, wbs, taus_b,  1)
    return exp_

def _phase_calc_four_point(beta, wsample, taus):
    """ Compute phase for Fourier transform of three-point obj in ph channel """
    wf1, wf2, wf3, wf4 = check_full_convention(*wsample)
    taus1, taus2, taus3, taus4 = taus
    exp_ = np.ones((wf1.size, taus1.size), dtype=np.complex128)
    exp_ *= _eval_exp(beta, wf1, taus1,  1)
    exp_ *= _eval_exp(beta, wf2, taus2, -1)
    exp_ *= _eval_exp(beta, wf3, taus3,  1)
    exp_ *= _eval_exp(beta, wf4, taus4, -1)
    return exp_

def ft_three_point_obj(worm_config_record, wsample, nflavors, beta):
    """ Compute three-point obj in ph channel """
    res = _ft_unique_freqs(worm_config_record, wsample, nflavors, beta, _phase_calc_three_point)
    res /= beta
    return res

def ft_four_point_obj(worm_config_record, wsample, nflavors, beta):
    """ Compute four-point obj in full convention """
    return _ft_unique_freqs(worm_config_record, wsample, nflavors, beta, _phase_calc_four_point)



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
    v_unique, v_where = np.unique(vsample, return_inverse=True)
    Tnl = compute_Tnl_sparse(v_unique, gl.shape[0])
    return np.einsum('wl,l...->w...', Tnl, gl)[v_where, :]


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
    """
    if exits_mc_result(h5, "Equal_time_G1_Re"):
        if verbose:
            print("Reading equal_time_G1...")
        results['equal_time_G1'] =  (w_vol/(sign * z_vol * beta)) * \
            read_cmplx_mc_result(h5, 'Equal_time_G1')['mean'].reshape((nflavors,nflavors))
    """

    return results

def postprocess_vartheta(h5, verbose=False, **kwargs):
    nflavors = kwargs['nflavors']
    beta = kwargs['beta']

    results = {}
    sign = read_mc_result(h5, 'Sign')['mean']
    w_vol = read_mc_result(h5, 'worm_space_volume_vartheta')['mean']
    z_vol = read_mc_result(h5, 'Z_function_space_volume')['mean']

    # vartheta
    if exits_mc_result(h5, "vartheta_legendre_Re"):
        if verbose:
            print("Reading vartheta_legendre...")
        results['vartheta_legendre'] =  -(w_vol/(beta * sign * z_vol)) * \
            read_cmplx_mc_result(h5, 'vartheta_legendre')['mean'].reshape((-1,nflavors,nflavors))

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

def postprocess_lambda(h5, verbose=False, **kwargs):
    nflavors = kwargs['nflavors']
    beta = kwargs['beta']
    sign = read_mc_result(h5, 'Sign')['mean']
    w_vol = read_mc_result(h5, 'worm_space_volume_lambda')['mean']
    z_vol = read_mc_result(h5, 'Z_function_space_volume')['mean']
    return {
        'lambda_legendre':
        (w_vol/(sign * z_vol * beta)) * \
            read_cmplx_mc_result(h5, 'lambda_legendre')['mean'].\
            reshape((-1,nflavors,nflavors,nflavors,nflavors))
    }

def postprocess_varphi(h5, verbose=False, **kwargs):
    nflavors = kwargs['nflavors']
    beta = kwargs['beta']
    sign = read_mc_result(h5, 'Sign')['mean']
    sign = read_mc_result(h5, 'Sign')['mean']
    w_vol = read_mc_result(h5, 'worm_space_volume_varphi')['mean']
    z_vol = read_mc_result(h5, 'Z_function_space_volume')['mean']
    if verbose:
        print("Reading varphi_legendre...")
    return {
        'varphi_legendre':
        (w_vol/(sign * z_vol * beta)) * \
            read_cmplx_mc_result(h5, 'varphi_legendre')['mean'].\
            reshape((-1,nflavors,nflavors,nflavors,nflavors))
    }

def postprocess_eta(h5, verbose=False, **kwargs):
    sign = read_mc_result(h5, 'Sign')['mean']
    w_vol = read_mc_result(h5, 'worm_space_volume_eta')['mean']
    z_vol = read_mc_result(h5, 'Z_function_space_volume')['mean']
    if verbose:
        print("Reading eta...")
    return {
        'eta_coeff' : w_vol/(sign * z_vol),
        'eta_datasets' :
            WormConfigRecord.load(kwargs['prefix'] + "_wormspace_eta_results", 3, 4)
    }

def postprocess_gamma(h5, verbose=False, **kwargs):
    sign = read_mc_result(h5, 'Sign')['mean']
    w_vol = read_mc_result(h5, 'worm_space_volume_gamma')['mean']
    z_vol = read_mc_result(h5, 'Z_function_space_volume')['mean']
    if verbose:
        print("Reading gamma...")
    return {
        'gamma_coeff' : w_vol/(sign * z_vol),
        'gamma_datasets' :
            WormConfigRecord.load(kwargs['prefix'] + "_wormspace_gamma_results", 3, 4)
    }

def postprocess_h(h5, verbose=False, **kwargs):
    if verbose:
        print("Reading h...")
    beta = kwargs['beta']
    prefix = kwargs['prefix']
    sign = read_mc_result(h5, 'Sign')['mean']
    w_vol = read_mc_result(h5, 'worm_space_volume_h')['mean']
    z_vol = read_mc_result(h5, 'Z_function_space_volume')['mean']
    return {
        'h_corr_coeff': w_vol/(sign * z_vol),
        'h_corr_datasets':
            WormConfigRecord.load(prefix + "_wormspace_h_results", 4, 4)
    }


postprocessors = {
    'G1'             : postprocess_G1,
    'vartheta'       : postprocess_vartheta,
    'Equal_time_G1'  : postprocess_equal_time_G1,
    'lambda'         : postprocess_lambda,
    'varphi'         : postprocess_varphi,
    #'G2'             : postprocess_G2,
    'eta'            : postprocess_eta,
    'gamma'          : postprocess_gamma,
    'h'             : postprocess_h,
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

    def compute_Delta_iv(self, wfs):
        wfs_unique, wfs_where = np.unique(wfs, return_inverse=True)
        return self.basis_f.evaluate_iw(self.Delta_l, wfs_unique)[wfs_where,...]

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
        return _einsum('abij,ij->ab', self.get_asymU(), self.get_dm())

    def compute_g0iv(self, wfs):
        return self._compute_non_int_giv(wfs, self.hopping)

    def _compute_non_int_giv(self, wfs, hopping):
        """ Compute non-interacting Green's function for given hopping matrix"""
        wfs = check_fermionic(wfs)
        wfs_unique, wfs_where = np.unique(wfs, return_inverse=True)
        
        # Compute Delta(iv)
        Delta_iv = self.compute_Delta_iv(wfs_unique)

        # Compute non-interacting Green's function
        G0 = np.empty((wfs_unique.size, self.nflavors, self.nflavors), dtype=np.complex128)
        I = np.identity(self.nflavors)
        for ifreq, v in enumerate(wfs_unique):
            iv = 1J * v * np.pi/self.beta
            G0[ifreq, ...] = np.linalg.inv(iv * I - Delta_iv[ifreq, ...] - hopping[None,:,:])
        return G0[wfs_where, ...]

    def compute_xi(self, wfs):
        """ Compute xi(iv) """
        wfs = check_fermionic(wfs)
        wfs_unique, wfs_where = np.unique(wfs, return_inverse=True)
        return _einsum(
            'wai,wib->wab',
            (self.compute_v()[None,:,:] + self.compute_vartheta(wfs_unique)),
            self.compute_g0iv(wfs_unique)
        )[wfs_where,...]
    
    def compute_phi(self, wbs):
        wbs = check_bosonic(wbs)
        wbs_unique, wbs_where = np.unique(wbs, return_inverse=True)
        return 0.25 * \
            _einsum(
                'abij,cdkl,wijkl->wabcd',
                self.get_asymU(),
                self.get_asymU(),
                self.compute_lambda(wbs_unique)
            )[wbs_where,...]

    def compute_Psi(self, wbs):
        wbs = check_bosonic(wbs)
        wbs_unique, wbs_where = np.unique(wbs, return_inverse=True)
        return 0.25 * \
            _einsum('ajci,kbld,Wijkl->Wabcd',
                self.get_asymU(), self.get_asymU(),
                self.compute_varphi(wbs_unique))[wbs_where,...]

    def compute_f(self, wfs, wbs):
        wfs = check_fermionic(wfs)
        wbs = check_bosonic(wbs)
        f = 0.5 * np.einsum(
            'Wabij,cdij->Wabcd',
            self.compute_eta(wfs, wbs),
            self.get_asymU(),
            optimize=True
        )
        return f
    
    def compute_g(self, wfs, wfs_p):
        wfs = check_fermionic(wfs)
        wfs_p = check_fermionic(wfs_p)
        return 0.5 * _einsum('jbkd,Wacjk->Wabcd',
           self.get_asymU(),
           self.compute_gamma(wfs, wfs + wfs_p)
        )
    
    def compute_scrF(self, wsample_full):
        wsample_full = _check_full_convention(*wsample_full)
        v1, v2, v3, v4 = wsample_full
        beta = self.beta
        asymU = self.get_asymU()
        scrF = np.zeros((wsample_full[0].size,) + 4*(self.nflavors,), dtype=np.complex128)

        vab = self.compute_v()

        scrF += beta * asymU[None, ...]

        t1 = time.time()
        print("p0 ", time.time()-t1)
        v1_ = self.compute_vartheta(v1) + vab[None,:,:]
        v3_ = self.compute_vartheta(v3) + vab[None,:,:]
        scrF += (beta**2) * _einsum('W,Wab,Wcd->Wabcd', v1==v2, v1_, v3_)
        scrF -= (beta**2) * _einsum('W,Wad,Wcb->Wabcd', v1==v4, v1_, v3_)
        print("p1 ", time.time()-t1)

        # xi
        scrF += beta * _einsum('ibcd,Wai->Wabcd', asymU, self.compute_xi(v1))
        scrF += beta * _einsum('aicd,Wbi->Wabcd', asymU, self.compute_xi(-v2).conj())
        scrF += beta * _einsum('abid,Wci->Wabcd', asymU, self.compute_xi(v3))
        scrF += beta * _einsum('abci,Wdi->Wabcd', asymU, self.compute_xi(-v4).conj())
        print("p2 ", time.time()-t1)

        # phi
        scrF += -4 * beta * self.compute_phi(v1-v2)
        scrF +=  4 * beta * _einsum('Wadcb->Wabcd', self.compute_phi(v1-v4))
        print("p3 ", time.time()-t1)

        # f
        scrF +=  2 * beta * self.compute_f(v1, v1-v2)
        scrF +=  2 * beta * _einsum('Wcdab->Wabcd', self.compute_f(v3, v2-v1))
        scrF += -2 * beta * _einsum('Wadcb->Wabcd', self.compute_f(v1, v1-v4))
        scrF += -2 * beta * _einsum('Wcbad->Wabcd', self.compute_f(v3, v4-v1))
        print("p4 ", time.time()-t1)

        # g
        scrF += -beta * self.compute_g(v1, v3)
        scrF += -beta * _einsum('Wdcba->Wabcd', self.compute_g(-v4, -v2).conj())
        print("p5 ", time.time()-t1)

        # Psi
        scrF += -beta * self.compute_Psi(v1+v3)
        print("p6 ", time.time()-t1)

        # h
        scrF -= self.compute_h(wsample_full)
        print("p7 ", time.time()-t1)

        return scrF

    def compute_F(self, wsample_full):
        scrF = self.compute_scrF(wsample_full)

        # Replace legs from G0 to G
        v1, v2, v3, v4 = wsample_full
        r1 = self._invG_G0(v1)
        r2 = self._G0_invG(v2)
        r3 = self._invG_G0(v3)
        r4 = self._G0_invG(v4)
        print("Calling compute_scrF...")
        F = _einsum('waA,wBb,wcC,wDd,wABCD->wabcd', r1, r2, r3, r4, scrF)
        return F

    def _invG_G0(self, wfs):
        wfs_unique, wfs_where = np.unique(wfs, return_inverse=True)
        res_unique = np.empty((wfs_unique.size, self.nflavors, self.nflavors), dtype=np.complex128)
        giv = self.compute_giv(wfs_unique)
        g0iv = self.compute_g0iv(wfs_unique)
        for i in range(wfs_unique.size):
            res_unique[i,:,:] = np.linalg.inv(giv[i,...]) @ g0iv[i,...]
        return res_unique[wfs_where, ...]

    def _G0_invG(self, wfs):
        wfs_unique, wfs_where = np.unique(wfs, return_inverse=True)
        res_unique = np.empty((wfs_unique.size, self.nflavors, self.nflavors), dtype=np.complex128)
        giv = self.compute_giv(wfs_unique)
        g0iv = self.compute_g0iv(wfs_unique)
        for i in range(wfs_unique.size):
            res_unique[i,:,:] = g0iv[i,...] @ np.linalg.inv(giv[i,...])
        return res_unique[wfs_where, ...]

    
    def compute_g4pt(self, wsample_full, F=None):
        wsample_full = _check_full_convention(*wsample_full)
        if F is None:
            F = self.compute_F(wsample_full)
        v1, v2, v3, v4 = wsample_full
        g1 = self.compute_giv(v1)
        g2 = self.compute_giv(v2)
        g3 = self.compute_giv(v3)
        g4 = self.compute_giv(v4)
        g4pt = (self.beta**2) * (
            _einsum('w,wab,wcd->wabcd', v1==v2, g1, g3)-_einsum('w,wad,wcb->wabcd', v1==v4, g1, g3)
        ) -_einsum('waA,wBb,wcC,wDd,wABCD->wabcd', g1, g2, g3, g4, F)
        return g4pt

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
        return self.compute_g0iv(wfs)

    def compute_vartheta(self, wfs):
        """ Compute vartheta(wfs) """
        wfs = check_fermionic(wfs)
        return np.zeros((wfs.size, self.nflavors, self.nflavors), dtype=np.complex128)

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
        return np.zeros((wfs.size,) + 4*(self.nflavors,), dtype=np.complex128)
    
    def compute_gamma(self, wfs, wbs):
        """ Compute gamma(wfs, wbs) """
        wfs = check_fermionic(wfs)
        wbs = check_bosonic(wbs)
        return np.zeros((wfs.size,) + 4*(self.nflavors,), dtype=np.complex128)

    def compute_h(self, wsample_full):
        """ Compute h(v1, v2, v3, v4) """
        v1, v2, v3, v4 = _check_full_convention(*wsample_full)
        return np.zeros((v1.size,) + 4*(self.nflavors,), dtype=np.complex128)

    def compute_g4pt(self, wsample_full):
        wsample_full = _check_full_convention(*wsample_full)
        v1, v2, v3, v4 = wsample_full
        g1 = self.compute_giv(v1)
        g3 = self.compute_giv(v3)
        g4pt = (self.beta**2) * (
            _einsum('w,wab,wcd->wabcd', v1==v2, g1, g3)-_einsum('w,wad,wcb->wabcd', v1==v4, g1, g3)
        ) 
        return g4pt

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
        self.ham_U = construct_ham(np.zeros_like(hopping), self.asymU, self.cdag_ops)
        self.evals, self.evecs = np.linalg.eigh(self.ham.toarray())
        self.q_ops = [c@self.ham_U-self.ham_U@c for c in self.c_ops]
        self.qdag_ops = [self.ham_U@cdag-cdag@self.ham_U for cdag in self.cdag_ops]

        self.dm = np.zeros((nflavors,nflavors), dtype=np.complex128)
        for i, j in product(range(nflavors), repeat=2):
            self.dm[i,j] = compute_expval(
                self.cdag_ops[i]@self.c_ops[j], beta, self.evals, self.evecs)

        self.vab = np.zeros((self.nflavors, self.nflavors), dtype=object)
        for a, b in product(range(self.nflavors), repeat=2):
            for i, j in product(range(self.nflavors), repeat=2):
                self.vab[a,b]+= self.asymU[a,b,i,j] * (self.cdag_ops[i]@self.c_ops[j])

    def get_asymU(self):
        return self.asymU

    def get_dm(self):
        return self.dm

    def compute_giv(self, wfs):
        wfs = check_fermionic(wfs)
        giv = np.empty((wfs.size,self.nflavors, self.nflavors), dtype=np.complex128)
        for i, j in product(range(self.nflavors), repeat=2):
            giv[:,i,j] = compute_fermionic_2pt_corr_func(
                self.c_ops[i], self.cdag_ops[j], self.beta, wfs, self.evals, self.evecs)
        return giv

    def compute_Delta_iv(self, wfs):
        return np.zeros((wfs.size, self.nflavors, self.nflavors), dtype=np.complex128)

    def compute_vartheta(self, wfs):
        """ Compute vartheta(wfs) """
        wfs = check_fermionic(wfs)
        vartheta = np.empty((wfs.size, self.nflavors, self.nflavors), dtype=np.complex128)
        for i, j in product(range(self.nflavors), repeat=2):
            vartheta[:,i,j] = compute_fermionic_2pt_corr_func(
                self.q_ops[i], self.qdag_ops[j], self.beta, wfs, self.evals, self.evecs)
        return vartheta

    def compute_phi(self, wbs):
        wbs = check_bosonic(wbs)
        phi = np.empty((wbs.size,)+ 4*(self.nflavors,), dtype=np.complex128)
        for i, j, k, l in product(range(self.nflavors),repeat=4):
            phi[:,i,j,k,l] = \
                0.25 * compute_bosonic_2pt_corr_func(
                    self.vab[i,j], self.vab[k,l], self.beta, wbs, self.evals, self.evecs)
        return phi                

    def compute_lambda(self, wbs):
        """ Compute lambda(wbs) """
        wbs = check_bosonic(wbs)
        lambda_wb = np.empty((wbs.size,)+ 4*(self.nflavors,), dtype=np.complex128)
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
        varphi_wb = np.empty((wbs.size,) + 4*(self.nflavors,), dtype=np.complex128)
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
        wslice = mpi.get_slice(wfs.size)
        wsample = (wfs[wslice], wbs[wslice])
        for i,j,k,l in product(range(self.nflavors), repeat=4):
            eta[wslice,i,j,k,l] = compute_3pt_corr_func(
                self.q_ops[i], self.qdag_ops[j], self.cdag_ops[k]@self.c_ops[l],
                self.beta, wsample, self.evals, self.evecs)
        return mpi.allreduce(eta)

    def compute_gamma(self, wfs, wbs):
        """ Compute eta(wfs, wbs) """
        wfs = check_fermionic(wfs)
        wbs = check_bosonic(wbs)
        gamma = np.zeros((wfs.size,)+(self.nflavors,)*4, dtype=np.complex128)
        wslice = mpi.get_slice(wfs.size)
        wsample = (wfs[wslice], wbs[wslice])
        for i,j,k,l in product(range(self.nflavors), repeat=4):
            gamma[wslice,i,j,k,l] = compute_3pt_corr_func(
                self.q_ops[i], self.q_ops[j], self.cdag_ops[k]@self.cdag_ops[l],
                self.beta, wsample, self.evals, self.evecs)
        return mpi.allreduce(gamma)
    
    def compute_h(self, wsample_full):
        wsample_full = _check_full_convention(*wsample_full)
        h = np.zeros((wsample_full[0].size,) + 4*(self.nflavors,), dtype=np.complex128)
        wslice = mpi.get_slice(wsample_full[0].size)
        wsample_full_local = tuple((v[wslice] for v in wsample_full))
        for i, j, k, l in product(range(self.nflavors),repeat=4):
            h[wslice,i,j,k,l] = compute_4pt_corr_func(
                self.q_ops[i], self.qdag_ops[j], self.q_ops[k], self.qdag_ops[l],
                self.beta, wsample_full_local, self.evals, self.evecs
            )
        return mpi.allreduce(h)
    
    def compute_g4pt_direct(self, wsample_full):
        """ Compute g4pt direct by ED """
        wsample_full = _check_full_convention(*wsample_full)
        g4pt = np.zeros((wsample_full[0].size,) + 4*(self.nflavors,), dtype=np.complex128)
        wslice = mpi.get_slice(wsample_full[0].size)
        wsample_full_local = tuple((v[wslice] for v in wsample_full))
        for i, j, k, l in product(range(self.nflavors),repeat=4):
            g4pt[wslice,i,j,k,l] = compute_4pt_corr_func(
                self.c_ops[i], self.cdag_ops[j], self.c_ops[k], self.cdag_ops[l],
                self.beta, wsample_full_local, self.evals, self.evecs
            )
        return mpi.allreduce(g4pt)

class VertexEvaluatorED(VertexEvaluator):
    """
    Exact diagonalization for
    """
    def __init__(self, beta, hopping_imp, asymU_imp, hopping_bath=None, hopping_coup=None):
        super().__init__()
        self.nflavors = hopping_imp.shape[0]
        if hopping_bath is None:
            self.nflavors_bath = 0
        else:
            self.nflavors_bath = hopping_bath.shape[0]
        self.nflavors_all = self.nflavors + self.nflavors_bath
        self.beta = beta
        self.hopping = hopping_imp
        self.asymU_imp = asymU_imp
        if self.nflavors_bath == 0:
            self.hopping_bath = np.zeros((self.nflavors_bath, self.nflavors_bath))
            self.hopping_coup = np.zeros((self.nflavors_bath, self.nflavors))
        self.hopping_bath = hopping_bath
        self.hopping_coup = hopping_coup

        self.hopping_all = np.block(
            [[hopping_imp, hopping_coup.T.conj()],
             [hopping_coup, hopping_bath]
             ])
        self.asymU = np.zeros(4*(self.nflavors_all,), dtype=np.complex128)
        self.asymU[
            0:self.nflavors,
            0:self.nflavors,
            0:self.nflavors,
            0:self.nflavors,
            ] = self.asymU_imp

        # Diagonalize the whole Hamiltonian
        nflavors = self.nflavors
        _, self.cdag_ops = construct_cdagger_ops(self.nflavors_all)
        self.c_ops = [op.transpose(copy=True) for op in self.cdag_ops]
        self.ham = construct_ham(self.hopping_all, self.asymU, self.cdag_ops)
        self.ham_U = construct_ham(np.zeros_like(self.hopping_all), self.asymU, self.cdag_ops)
        self.evals, self.evecs = np.linalg.eigh(self.ham.toarray())
        self.q_ops = [c@self.ham_U-self.ham_U@c for c in self.c_ops]
        self.qdag_ops = [self.ham_U@cdag-cdag@self.ham_U for cdag in self.cdag_ops]

        self.dm = np.zeros((nflavors,nflavors), dtype=np.complex128)
        for i, j in product(range(nflavors), repeat=2):
            self.dm[i,j] = compute_expval(
                self.cdag_ops[i]@self.c_ops[j], beta, self.evals, self.evecs)

        self.vab = np.zeros((nflavors, nflavors), dtype=object)
        for a, b in product(range(nflavors), repeat=2):
            for i, j in product(range(nflavors), repeat=2):
                self.vab[a,b]+= self.asymU[a,b,i,j] * (self.cdag_ops[i]@self.c_ops[j])
        
        # Impurity (local) Hamiltonian
        _, self.cdag_ops_imp = construct_cdagger_ops(nflavors)
        self.ham = construct_ham(self.hopping, self.asymU_imp, self.cdag_ops_imp)
        self.evals_imp, self.evecs_imp = np.linalg.eigh(self.ham.toarray())

    def get_asymU(self):
        return self.asymU_imp

    def get_dm(self):
        return self.dm

    def compute_giv(self, wfs):
        wfs = check_fermionic(wfs)
        giv = np.empty((wfs.size,self.nflavors, self.nflavors), dtype=np.complex128)
        for i, j in product(range(self.nflavors), repeat=2):
            giv[:,i,j] = compute_fermionic_2pt_corr_func(
                self.c_ops[i], self.cdag_ops[j], self.beta, wfs, self.evals, self.evecs)
        return giv

    def compute_Delta_iv(self, wfs):
        Delta_iv = np.empty((wfs.size, self.nflavors, self.nflavors), dtype=np.complex128)
        I = np.identity(self.nflavors)
        for ifreq, v in enumerate(wfs):
            iv = 1J * v * np.pi/self.beta
            inv = np.linalg.inv(iv * I - self.hopping_bath)
            Delta_iv[ifreq, ...] = _einsum('ia,ab,bj', self.hopping_coup.T.conj(), inv, self.hopping_coup)
        return Delta_iv

    def compute_vartheta(self, wfs):
        """ Compute vartheta(wfs) """
        wfs = check_fermionic(wfs)
        vartheta = np.empty((wfs.size, self.nflavors, self.nflavors), dtype=np.complex128)
        for i, j in product(range(self.nflavors), repeat=2):
            vartheta[:,i,j] = compute_fermionic_2pt_corr_func(
                self.q_ops[i], self.qdag_ops[j], self.beta, wfs, self.evals, self.evecs)
        return vartheta

    def compute_phi(self, wbs):
        wbs = check_bosonic(wbs)
        phi = np.empty((wbs.size,)+ 4*(self.nflavors,), dtype=np.complex128)
        for i, j, k, l in product(range(self.nflavors),repeat=4):
            phi[:,i,j,k,l] = \
                0.25 * compute_bosonic_2pt_corr_func(
                    self.vab[i,j], self.vab[k,l], self.beta, wbs, self.evals, self.evecs)
        return phi                

    def compute_lambda(self, wbs):
        """ Compute lambda(wbs) """
        wbs = check_bosonic(wbs)
        lambda_wb = np.empty((wbs.size,)+ 4*(self.nflavors,), dtype=np.complex128)
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
        varphi_wb = np.empty((wbs.size,) + 4*(self.nflavors,), dtype=np.complex128)
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
        wslice = mpi.get_slice(wfs.size)
        wsample = (wfs[wslice], wbs[wslice])
        for i,j,k,l in product(range(self.nflavors), repeat=4):
            eta[wslice,i,j,k,l] = compute_3pt_corr_func(
                self.q_ops[i], self.qdag_ops[j], self.cdag_ops[k]@self.c_ops[l],
                self.beta, wsample, self.evals, self.evecs)
        return mpi.allreduce(eta)

    def compute_gamma(self, wfs, wbs):
        """ Compute eta(wfs, wbs) """
        wfs = check_fermionic(wfs)
        wbs = check_bosonic(wbs)
        gamma = np.zeros((wfs.size,)+(self.nflavors,)*4, dtype=np.complex128)
        wslice = mpi.get_slice(wfs.size)
        wsample = (wfs[wslice], wbs[wslice])
        for i,j,k,l in product(range(self.nflavors), repeat=4):
            gamma[wslice,i,j,k,l] = compute_3pt_corr_func(
                self.q_ops[i], self.q_ops[j], self.cdag_ops[k]@self.cdag_ops[l],
                self.beta, wsample, self.evals, self.evecs)
        return mpi.allreduce(gamma)
    
    def compute_h(self, wsample_full):
        wsample_full = _check_full_convention(*wsample_full)
        h = np.zeros((wsample_full[0].size,) + 4*(self.nflavors,), dtype=np.complex128)
        wslice = mpi.get_slice(wsample_full[0].size)
        wsample_full_local = tuple((v[wslice] for v in wsample_full))
        for i, j, k, l in product(range(self.nflavors),repeat=4):
            h[wslice,i,j,k,l] = compute_4pt_corr_func(
                self.q_ops[i], self.qdag_ops[j], self.q_ops[k], self.qdag_ops[l],
                self.beta, wsample_full_local, self.evals, self.evecs
            )
        return mpi.allreduce(h)
    
    def compute_g4pt_direct(self, wsample_full):
        """ Compute g4pt direct by ED """
        wsample_full = _check_full_convention(*wsample_full)
        g4pt = np.zeros((wsample_full[0].size,) + 4*(self.nflavors,), dtype=np.complex128)
        wslice = mpi.get_slice(wsample_full[0].size)
        wsample_full_local = tuple((v[wslice] for v in wsample_full))
        for i, j, k, l in product(range(self.nflavors),repeat=4):
            g4pt[wslice,i,j,k,l] = compute_4pt_corr_func(
                self.c_ops[i], self.cdag_ops[j], self.c_ops[k], self.cdag_ops[l],
                self.beta, wsample_full_local, self.evals, self.evecs
            )
        return mpi.allreduce(g4pt)

class QMCResult(VertexEvaluator):
    def __init__(self, p, verbose=False, Lambda=1E+5, cutoff=1e-12) -> None:
        if verbose:
            print(p+'.out.h5')
    
        with h5py.File(p+'.out.h5','r') as h5:
            self.beta = read_param(h5, 'model.beta')
            self.nflavors = read_param(h5, 'model.flavors')
            self.sites = self.nflavors//2

        with h5py.File(p+'_wormspace_G1.out.h5','r') as h5:
            self.hopping = load_cmplx(h5, 'hopping')
            self.Delta_tau = load_cmplx(h5, '/Delta_tau').transpose((2,0,1))
            # U_tensor: (1/2) U_{ijkl} d^dagger_i d^dagger_j d_k d_l
            self.U_tensor = load_cmplx(h5, 'U_tensor')
            # asymU: (1/4) U_{ikjl} d^dagger_i d^dagger_j d_l d_k
            self.asymU = 2.0 * mk_asymm(self.U_tensor.copy().transpose(0, 3, 1, 2))
        
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
        # We interpolate Delta(tau) and evaluate it on sampling points to the following problem:
        #  If ntau is not large, we do not have data points sufficiently near tau=0, beta.
        #  This will increase the condition number of fitting matrix significantly.
        from scipy.interpolate import interp1d

        Delta_tau_interp = interp1d(np.linspace(0, self.beta, self.Delta_tau.shape[0]),
            self.Delta_tau, axis=0)

        taus_sp = self.basis_f.sampling_points_tau(self.basis_f.dim()-1)
        Delta_tau_sp = Delta_tau_interp(taus_sp)
        all_l = np.arange(self.basis_f.dim())
        Ftau = self.basis_f.Ultau(all_l[:,None], taus_sp[None,:]).T
        regularizer = self.basis_f.Sl(all_l)
        tol = self.basis_f.Sl(self.basis_f.dim()-1)/self.basis_f.Sl(0)
        self.Delta_l = _stable_fit(Ftau*regularizer[None,:], Delta_tau_sp, tol)
        self.Delta_l *= regularizer[:,None,None]
        self.Delta_tau_rec = np.einsum('tl,lij->tij', Ftau, self.Delta_l)

        self.evalU0 = VertexEvaluatorU0(
            self.nflavors, self.beta, self.basis_f, self.basis_b, self.hopping, self.Delta_l)

    def get_asymU(self):
        return self.asymU

    def get_dm(self):
        return self.equal_time_G1

    def compute_giv(self, wfs):
        return self.compute_giv_SIE(wfs)

    def compute_Delta_tau(self, taus):
        all_l = np.arange(self.basis_f.dim())
        Ftau = self.basis_f.Ultau(all_l[:,None], taus[None,:]).T
        return _einsum('tl,lij->tij', Ftau, self.Delta_l)

    def compute_gir_SIE(self):
        """
        Reconstruct one-particle Green's function using SIE
        in fermionic IR
        """
        vsample = self.basis_f.wsample
        giv = self.compute_giv_SIE(vsample)
        return _fit_iw(self.basis_f, giv)

    def compute_giv_SIE(self, vsample):
        """
        Reconstruct one-particle Green's function using SIE
        on Matsubara frequencies
        """
        vsample = check_fermionic(vsample)
        nfreqs = vsample.size
        
        # Compute vartheta
        vartheta = legendre_to_matsubara(self.vartheta_legendre, vsample)

        # Compute A_{ab}
        A = np.einsum('abij,ij->ab', self.asymU, self.equal_time_G1)

        # Compute G0, full G, self-energy
        # FIXME: Compute self-energy directry
        G = np.empty((nfreqs, self.nflavors, self.nflavors), dtype=np.complex128)
        G0 = self.compute_g0iv(vsample)
        for ifreq, v in enumerate(vsample):
            G0_ = G0[ifreq,:,:]
            G[ifreq, ...] = G0_ + \
                np.einsum('ai,ij,jb->ab', G0_,  (A + vartheta[ifreq,...]), G0_, optimize=True)
        
        return G 
    
    def compute_sigma_iv(self, giv, vsample):
        vsample = check_fermionic(vsample)
        xi = self.compute_xi(vsample)
        Sigma = np.empty((vsample.size, self.nflavors, self.nflavors), dtype=np.complex128)
        for ifreq, _ in enumerate(vsample):
            Sigma[ifreq, ...] = xi[ifreq,:,:] @ np.linalg.inv(giv[ifreq,:,:])
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
        if not hasattr(self, "gamma_coeff"):
            return np.zeros((wfs.size,) + 4*(self.nflavors,), dtype=np.complex128)
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
    
def reconst_vartheta(asymU, giv, g0iv, dm):
    """
    Reconstruct vartheta from asymU, giv, gi0v, and dm
    """
    #print("giv", g0iv[0,:,:])
    #print("g0iv", g0iv[0,:,:])
    #print("asymU", asymU)
    #print("dm", dm)
    nfreqs = giv.shape[0]
    v = np.einsum('abij,ij->ab', asymU, dm)
    vartheta = np.zeros_like(giv)
    for ifreq in range(nfreqs):
        inv_g0iv = np.linalg.inv(g0iv[ifreq,:,:])
        vartheta[ifreq,:,:] = inv_g0iv @ (giv[ifreq,:,:] - g0iv[ifreq,:,:]) @ inv_g0iv - v
    #print("vartheta", vartheta[0,], )
    return vartheta
