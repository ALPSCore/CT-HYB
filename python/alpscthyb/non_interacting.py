import numpy as np
from alpscthyb.post_proc import QMCResult
from alpscthyb import post_proc
from irbasis_x.freq import check_fermionic, check_bosonic, check_full_convention


class NoninteractingLimit:
    def __init__(self, res) -> None:
        """
        res: QMCResult
        """
        self._res = res

        self.basis_f = res.basis_f
        self.nflavors = res.nflavors
        self.beta = res.beta
        self.hopping = res.hopping

        self.g0ir = post_proc._fit_iw(
            self.basis_f,
            self.giv(self.basis_f.wsample)
        )

        """
        n_{ab} = <d^\dagger_a d_b>
        """
        self.dm = -self.gtau(self.beta).T.reshape(2*(self.nflavors,))

        """
        v_{ab}
        """
        self.vab = res.hopping + 0.5 * np.einsum('abij,ij->ab', res.asymU, self.dm)
    
    def giv(self, wfsample):
        return self._giv(wfsample, self.hopping)

    def calgiv(self, wfsample):
        return self._giv(wfsample, np.zeros(2*(self.nflavors,)))

    def _giv(self, wfsample, hopping):
        wfsample = check_fermionic(wfsample)
        nfreqs = wfsample.size
        
        # Compute Delta(iv)
        Delta_iv = self.basis_f.evaluate_iw(self._res.Delta_l, wfsample)

        # Compute non-interacting Green's function
        G0 = np.empty((nfreqs, self.nflavors, self.nflavors), dtype=np.complex128)
        I = np.identity(self.nflavors)
        for ifreq, v in enumerate(wfsample):
            iv = 1J * v * np.pi/self.beta
            G0[ifreq, ...] = np.linalg.inv(
                iv * I - Delta_iv[ifreq, ...] - hopping[None,:,:])
        return G0
    
    def gtau(self, taus):
        taus = np.atleast_1d(taus)
        assert all(np.array(taus >= 0.0, ndmin=1))
        assert all(np.array(taus <= self.beta, ndmin=1))
        return np.einsum('tl,lij->tij', self.basis_f.Ultau_all_l(taus).T, self.g0ir)

    def lambda_tau(self, tau):
        """ Compute lambda(tau) """
        res = self._res
        gtau     = self.gtau(tau)
        gtau_inv = self.gtau(res.beta-tau)
        return np.einsum('ab,cd->abcd', self.dm, self.dm)[None,...] + \
            np.einsum('Tda, Tbc->Tabcd', gtau_inv, gtau)

    def varphi_tau(self, tau):
        """ Compute lambda(tau) """
        res = self._res
        gtau = np.einsum('tl,lij->tij', res.basis_f.Ultau_all_l(tau).T, self.g0ir)
        return np.einsum('Tad,Tbc->Tabcd', gtau, gtau) - np.einsum('Tac,Tbd->Tabcd', gtau, gtau)

    def vartheta(self, wfsample):
        wfsample = check_fermionic(wfsample)
        return np.einsum('ai,jb,wij->wab', self._res.hopping,
            self.hopping, self.giv(wfsample))

    def xi_iv(self, wfsample):
        """ Compute xi(iv) """
        return np.einsum(
            'wai,wib->wab',
            self.calgiv(wfsample),
            (self.vab[None,:,:] + self.vartheta(wfsample))
        )

    def eta(self, wsample):
        """ Compute eta(wf, wb) """
        wfsample, wbsample = wsample
        assert wfsample.size == wbsample.size
        wfsample = check_fermionic(wfsample)
        wbsample = check_bosonic(wbsample)

        eta1 = - self.beta * \
           np.einsum('w,wab,cd->wabcd',
               (wbsample == 0),
               self.vartheta(wfsample),
               self.dm
           )
        eta2 = np.einsum('wac,wbd->wabcd',
           self.xi_iv(wfsample),
           self.xi_iv(wbsample-wfsample).conj()
        )
        return eta1 + eta2

    def gamma(self, wsample):
        """ Compute gamma(wf, wb) """
        wfsample, wbsample = wsample
        assert wfsample.size == wbsample.size
        wfsample = check_fermionic(wfsample)
        wbsample = check_bosonic(wbsample)
        xi1 = self.xi_iv(wfsample)
        xi2 = self.xi_iv(wbsample-wfsample)
        return np.einsum('wad,wbc->wabcd', xi1, xi2) - np.einsum('wac,wbd->wabcd', xi1, xi2)
    
    def h(self, wsample):
        """ Compute h(v1, v2, v3, v4) """
        v1, v2, v3, v4 = check_full_convention(*wsample)
        vartheta1 = self.vartheta(v1)
        vartheta3 = self.vartheta(v3)
        print("v1", v1)
        print("v2", v2)
        print("v3", v3)
        print("v4", v4)
        print("v1==v2", v1==v2)
        print("v1==v3", v1==v3)
        return (self.beta**2) * (
            np.einsum('w,wab,wcd->wabcd', (v1==v2), vartheta1, vartheta3) -
            np.einsum('w,wad,wcb->wabcd', (v1==v3), vartheta1, vartheta3)
        )