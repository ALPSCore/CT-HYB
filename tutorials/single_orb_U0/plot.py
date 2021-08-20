import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from irbasis_x.freq import box
from numpy.core.einsumfunc import einsum
from alpscthyb import non_interacting
from alpscthyb.post_proc import QMCResult, compute_Tnl_sparse, legendre_to_matsubara, legendre_to_tau
from alpscthyb import post_proc
from alpscthyb.non_interacting import NoninteractingLimit


res = QMCResult('input', verbose=True)
non_int = NoninteractingLimit(res)
beta = res.beta

plt.figure(1)
for f in range(res.nflavors):
    plt.plot(res.Delta_tau[:,f,f], label=f'flavor{f}')
    plt.plot(res.Delta_tau_rec[:,f,f], label=f'flavor{f}')
plt.legend()
plt.savefig("Delta_tau.eps")
plt.close(1)

plt.figure(1)
for f in range(res.nflavors):
    plt.semilogy(np.abs(res.Delta_l[:,f,f]), label=f'flavor{f}')
plt.legend()
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
g0iv = non_int.giv(res.basis_f.wsample)

# vartheta
#vartheta = res.vartheta
vartheta_rec = legendre_to_matsubara(res.vartheta_legendre, vsample)
vartheta_non_int = non_int.vartheta(vsample)

v = vsample * np.pi/res.beta
iv = 1J * v

for flavor in range(res.nflavors):
    plt.figure(1)
    plt.subplot(211)
    v_ = vsample * np.pi/beta
    plt.plot(v_, vartheta_rec[:,flavor,flavor].real, ls='', marker='x', label='SIE (legendre)')
    plt.plot(v_, vartheta_non_int[:,flavor,flavor].real, ls='', marker='+', label='Non interacting')
    plt.xlim([-10, 10])
    plt.ylabel(r"Re$\vartheta(\mathrm{i}\nu)$")
    plt.subplot(212)
    plt.plot(v_, vartheta_rec[:,flavor,flavor].imag, ls='', marker='x', label='SIE (legendre)')
    plt.plot(v_, vartheta_non_int[:,flavor,flavor].imag, ls='', marker='+', label='Non interacting')
    plt.xlim([-10, 10])
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"Im$\vartheta(\mathrm{i}\nu)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"vartheta_flavor{flavor}.eps")
    plt.close(1)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(v, giv[:,flavor,flavor].real, label='SIE')
    plt.plot(v, giv_legendre[:,flavor,flavor].real, label='Legendre')
    plt.plot(v, g0iv[:,flavor,flavor].real, label='G0')
    plt.xlim([-10, 10])
    plt.ylabel(r"Re$G(\mathrm{i}\nu)$")
    plt.subplot(212)
    plt.plot(v, giv[:,flavor,flavor].imag, label='SIE')
    plt.plot(v, giv_legendre[:,flavor,flavor].imag, marker='x', ls='', label='Legendre')
    plt.plot(v, g0iv[:,flavor,flavor].imag, label='G0')
    plt.xlim([-10, 10])
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"Im$G(\mathrm{i}\nu)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"giv_flavor{flavor}.eps")
    plt.close(1)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(v, (sigma_iv[:,flavor,flavor].real), label='SIE')
    plt.plot(v, (sigma_iv_legendre[:,flavor,flavor].real), marker='x', ls='', label='legenre')
    plt.ylabel(r"Re$\Sigma(\mathrm{i}\nu)$")
    plt.subplot(212)
    plt.plot(v, (sigma_iv[:,flavor,flavor].imag), label='SIE')
    plt.plot(v, (sigma_iv_legendre[:,flavor,flavor].imag), marker='x', ls='', label='legenre')
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"Im$\Sigma(\mathrm{i}\nu)$")
    plt.legend()
    plt.savefig(f"sigma_iv_flavor{flavor}.eps")
    plt.close(1)


ref_generators = {'lambda': 'lambda_tau', 'varphi': 'varphi_tau'}
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
        plt.plot(tau, (data_tau[:,flavors[0],flavors[1],flavors[2],flavors[3]].real), label='Re', ls='', marker='x', color='r')
        plt.plot(tau, (data_tau[:,flavors[0],flavors[1],flavors[2],flavors[3]].imag), label='Im', ls='', marker='x', color='b')
        if name in ref_generators:
            data_tau_ref = non_int.__getattribute__(ref_generators[name])(tau)
            plt.plot(tau, (data_tau_ref[:,flavors[0],flavors[1],flavors[2],flavors[3]].real), ls='--', marker='', color='r')
            plt.plot(tau, (data_tau_ref[:,flavors[0],flavors[1],flavors[2],flavors[3]].imag), ls='--', marker='', color='b')
        plt.legend()
        plt.savefig(name + f"_tau_flavors{flavors[0]}{flavors[1]}{flavors[2]}{flavors[3]}.eps")
        plt.close(1)


def box_fb(nf, nb):
    wf = 2*np.arange(-nf,nf)+1
    wb = 2*np.arange(-nb,nb)
    v, w = np.broadcast_arrays(wf[:,None], wb[None,:])
    return v.ravel(), w.ravel()

wsample_fb = box_fb(4, 5)

# eta
fig, axes = plt.subplots(2, 1)
eta_qmc = res.compute_eta(wsample_fb)
eta_ref = non_int.eta(wsample_fb)
axes[0].plot(eta_qmc[:,0,0,0,0].real, marker='o', label='QMC')
axes[0].plot(eta_ref[:,0,0,0,0].real, marker='x', label='ref')
axes[1].plot(eta_qmc[:,0,0,0,0].imag, marker='o', label='QMC')
axes[1].plot(eta_ref[:,0,0,0,0].imag, marker='x', label='ref')
axes[0].set_ylabel(r"Re$\eta$")
axes[1].set_ylabel(r"Im$\eta$")
for ax in axes:
    ax.legend()
fig.tight_layout()
fig.savefig('eta.eps')

# gamma
fig, axes = plt.subplots(2, 1)
gamma_qmc = res.compute_gamma(wsample_fb)
gamma_ref = non_int.gamma(wsample_fb)
axes[0].plot(gamma_qmc[:,0,1,0,1].real, marker='o', label='QMC')
axes[0].plot(gamma_ref[:,0,1,0,1].real, marker='x', label='ref')
axes[1].plot(gamma_qmc[:,0,1,0,1].imag, marker='o', label='QMC')
axes[1].plot(gamma_ref[:,0,1,0,1].imag, marker='x', label='ref')
axes[0].set_ylabel(r"Re$\gamma$")
axes[1].set_ylabel(r"Im$\gamma$")
for ax in axes:
    ax.legend()
fig.tight_layout()
fig.savefig('gamma.eps')

# h
wsample_ffff = box(4, 3, return_conv='full', ravel=True)
fig, axes = plt.subplots(2, 1)
h_qmc = res.compute_h_corr(wsample_ffff)
h_ref = non_int.h(wsample_ffff)
data = np.max(np.abs(h_ref), axis=0)
axes[0].plot(h_qmc[:,0,0,1,1].real, marker='o', label='QMC')
axes[0].plot(h_ref[:,0,0,1,1].real, marker='x', label='ref')
axes[1].plot(h_qmc[:,0,0,1,1].imag, marker='o', label='QMC')
axes[1].plot(h_ref[:,0,0,1,1].imag, marker='x', label='ref')
axes[0].set_ylabel(r"Re$h$")
axes[1].set_ylabel(r"Im$h$")
for ax in axes:
    ax.legend()
fig.tight_layout()
fig.savefig('h.eps')
