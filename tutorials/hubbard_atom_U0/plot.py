import numpy as np
import matplotlib.pyplot as plt
from numpy.core.einsumfunc import einsum
from alpscthyb.post_proc import QMCResult, compute_Tnl_sparse, legendre_to_matsubara, legendre_to_tau
from alpscthyb import post_proc


def lambda_tau_non_int(res, tau, gir):
    """ Compute lambda(tau) from gl in a non-interacting case"""
    #gtau = legendre_to_tau(gl, tau, beta)
    #gtau_inv = legendre_to_tau(gl, res.beta-tau, beta)
    gtau     = np.einsum('tl,lij->tij', res.basis_f.Ultau_all_l(tau).T,          gir)
    gtau_inv = np.einsum('tl,lij->tij', res.basis_f.Ultau_all_l(res.beta-tau).T, gir)
    #gtau = res.basis_f.evaluate_tau(gir)
    #gtau = res.basis_f.evaluate_tau(gir)
    return np.einsum('ab,cd->abcd', res.equal_time_G1, res.equal_time_G1)[None,...] + \
        np.einsum('Tda, Tbc->Tabcd', gtau_inv, gtau)


def varphi_tau_non_int(res, tau, gir):
    """ Compute lambda(tau) from gl in a non-interacting case"""
    gtau = np.einsum('tl,lij->tij', res.basis_f.Ultau_all_l(tau).T,          gir)
    return np.einsum('Tad,Tbc->Tabcd', gtau, gtau) - np.einsum('Tac,Tbd->Tabcd', gtau, gtau)


#def giw_ref(beta, U, vsample):
    #assert all(vsample%2 == 1)
    #iv = 1J * vsample * np.pi/beta
    #return 0.5/(iv-0.5*U) + 0.5/(iv+0.5*U)
#
#def vartheta_ref(beta, U, vsample):
    #return ((0.5*U)**2) * giw_ref(beta, U, vsample)

res = QMCResult('input')
beta = res.beta

plt.figure(1)
plt.semilogy(np.abs(res.Delta_tau[0,0,:]))
plt.savefig("Delta_tau.eps")
plt.close(1)

plt.figure(1)
plt.semilogy(np.abs(res.Delta_l[:,0,0]))
plt.savefig("Delta_l.eps")
plt.close(1)

U = 8.
mu = 0.5*U

# Fermionic sampling frequencies
vsample = res.basis_f.wsample

#SIE
gir_SIE = res.compute_gir_SIE()
giv = res.compute_giv_SIE(vsample)
sigma_iv = res.compute_sigma_iv(giv, vsample)

#Legendre
giv_legendre = res.compute_giv_from_legendre(vsample)
sigma_iv_legendre = res.compute_sigma_iv(giv_legendre, vsample)

# G0
g0iv = res.compute_g0iv(res.basis_f.wsample)
g0ir = post_proc._fit_iw(res.basis_f, g0iv)

v = vsample * np.pi/res.beta
iv = 1J * v

#giv_ref = 0.5/(iv - 0.5*U) + 0.5/(iv + 0.5*U)
#g0      = 1/iv
#sigma_iv_ref = 1/g0 - 1/giv_ref + mu

for flavor in range(res.nflavors):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(v, giv[:,flavor,flavor].real, label='SIE')
    plt.plot(v, giv_legendre[:,flavor,flavor].real, label='Legendre')
    plt.xlim([-20, 20])
    plt.ylabel(r"Re$G(\mathrm{i}\nu)$")
    plt.subplot(212)
    plt.plot(v, giv[:,flavor,flavor].imag, label='SIE')
    plt.plot(v, giv_legendre[:,flavor,flavor].imag, marker='x', ls='', label='Legendre')
    plt.xlim([-20, 20])
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"Im$G(\mathrm{i}\nu)$")
    plt.legend()
    plt.savefig(f"giv_flavor{flavor}.eps")
    plt.close(1)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(v, (sigma_iv[:,flavor,flavor].real), label='SIE')
    plt.plot(v, (sigma_iv_legendre[:,flavor,flavor].real), marker='x', ls='', label='legenre')
    #plt.plot(v, np.abs(sigma_iv_ref.real), marker='+', ls='', label='ref')
    plt.ylabel(r"Re$\Sigma(\mathrm{i}\nu)$")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.subplot(212)
    plt.plot(v, (sigma_iv[:,flavor,flavor].imag), label='SIE')
    plt.plot(v, (sigma_iv_legendre[:,flavor,flavor].imag), marker='x', ls='', label='legenre')
    #plt.plot(v, np.abs(sigma_iv_ref.imag), marker='+', ls='', label='ref')
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"Im$\Sigma(\mathrm{i}\nu)$")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    plt.savefig(f"sigma_iv_flavor{flavor}.eps")
    plt.close(1)


ref_generators = {'lambda': lambda_tau_non_int, 'varphi': varphi_tau_non_int}
for name in ['varphi', 'lambda']:
    data_l = res.__getattribute__(name+'_legendre')
    wsample = 2 * np.arange(-10,10)
    data_iw = legendre_to_matsubara(data_l, wsample)
    tau = np.linspace(0, beta, 100)
    data_tau = legendre_to_tau(data_l, tau, beta)
    for flavors in [(0,1,0,1), (1,0,1,0), (0,0,0,0), (1,1,1,1)]:
        plt.figure(1)
        plt.plot(wsample, (data_iw[:,flavors[0],flavors[1],flavors[2],flavors[3]].real), label='Re', marker='x')
        plt.plot(wsample, (data_iw[:,flavors[0],flavors[1],flavors[2],flavors[3]].imag), label='Im', marker='x')
        plt.legend()
        plt.xlim([-10,10])
        plt.savefig(name + f"_iw_flavors{flavors[0]}{flavors[1]}{flavors[2]}{flavors[3]}.eps")
        plt.close(1)

        plt.figure(1)
        plt.plot(tau, (data_tau[:,flavors[0],flavors[1],flavors[2],flavors[3]].real), label='Re', marker='x', color='r')
        plt.plot(tau, (data_tau[:,flavors[0],flavors[1],flavors[2],flavors[3]].imag), label='Im', marker='x', color='b')
        if name in ref_generators:
            data_tau_ref = ref_generators[name](res, tau, g0ir)
            plt.plot(tau, (data_tau_ref[:,flavors[0],flavors[1],flavors[2],flavors[3]].real), ls='-', marker='', color='r')
            plt.plot(tau, (data_tau_ref[:,flavors[0],flavors[1],flavors[2],flavors[3]].imag), ls='-', marker='', color='b')
        plt.legend()
        plt.savefig(name + f"_tau_flavors{flavors[0]}{flavors[1]}{flavors[2]}{flavors[3]}.eps")
        plt.close(1)


#plt.figure(1)
#plt.plot(v, np.abs(sigma_iv[:,0,0] - sigma_iv_ref), marker='x', label='SIE')
#plt.plot(v, np.abs(sigma_iv_legendre[:,0,0] - sigma_iv_ref), marker='+', label='Legendre')
#plt.xlabel(r"$\nu$")
#plt.ylabel(r"Absolute error in $\Sigma(\mathrm{i}\nu)$")
#plt.xscale("log")
#plt.yscale("log")
#plt.legend()
#plt.savefig("error_sigma_iv.eps")
#plt.close(1)
