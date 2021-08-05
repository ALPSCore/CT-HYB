import numpy as np
import matplotlib.pyplot as plt
from numpy.core.einsumfunc import einsum
from alpscthyb.post_proc import QMCResult, compute_Tnl_sparse

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

#SIE
vsample = res.vartheta_smpl_freq
giv = res.compute_giv_SIE()
sigma_iv = res.compute_sigma_iv(giv, vsample)

#Legendre
giv_legendre = res.compute_giv_from_legendre(vsample)
sigma_iv_legendre = res.compute_sigma_iv(giv_legendre, vsample)

v = res.vartheta_smpl_freq * np.pi/res.beta
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


for name in ['varphi', 'lambda']:
    data_l = res.__getattribute__(name+'_legendre')
    data_iv = np.einsum('wl,labcd->wabcd',
        compute_Tnl_sparse(vsample, data_l.shape[0]),
        data_l
    )
    for flavors in [(0,1,0,1), (0,0,0,0)]:
        plt.figure(1)
        plt.plot(vsample, (data_iv[:,flavors[0],flavors[1],flavors[2],flavors[3]].real), label='Re')
        plt.plot(vsample, (data_iv[:,flavors[0],flavors[1],flavors[2],flavors[3]].imag), label='Im')
        plt.legend()
        plt.xlim([-10,10])
        plt.savefig(name + f"_iv_flavors{flavors[0]}{flavors[1]}{flavors[2]}{flavors[3]}.eps")
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
