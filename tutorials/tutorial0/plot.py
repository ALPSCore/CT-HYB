import numpy as np
import matplotlib.pyplot as plt
from alpscthyb.post_proc import QMCResult

def giw_ref(beta, U, vsample):
    assert all(vsample%2 == 1)
    iv = 1J * vsample * np.pi/beta
    return 0.5/(iv-0.5*U) + 0.5/(iv+0.5*U)

def vartheta_ref(beta, U, vsample):
    return ((0.5*U)**2) * giw_ref(beta, U, vsample)

res = QMCResult('input')
beta = res.beta

plt.figure(1)
plt.semilogy(np.abs(res.Delta_tau[:,0,0]))
plt.savefig("Delta_tau.pdf")
plt.close(1)

plt.figure(1)
plt.semilogy(np.abs(res.Delta_l[:,0,0]))
plt.savefig("Delta_l.pdf")
plt.close(1)

#SIE
vsample = res.vartheta_smpl_freq
giv = res.compute_giv_SIE()
sigma_iv = res.compute_sigma_iv(giv, vsample)

#Legendre
giv_legendre = res.compute_giv_from_legendre(vsample)
sigma_iv_legendre = res.compute_sigma_iv(giv_legendre, vsample)

v = res.vartheta_smpl_freq * np.pi/res.beta
iv = 1J * v

plt.figure(1)
plt.subplot(211)
plt.plot(v, res.vartheta[:,0,0].real, ls='-', marker='x', label='SIE')
plt.xlim([-100, 100])
plt.ylabel(r"Re$\vartheta(\mathrm{i}\nu)$")
plt.subplot(212)
plt.plot(v, res.vartheta[:,0,0].imag, ls='-', marker='x', label='SIE')
plt.xlim([-100, 100])
plt.xlabel(r"$\nu$")
plt.ylabel(r"Im$\vartheta(\mathrm{i}\nu)$")
plt.legend()
plt.savefig("vartheta.pdf")
plt.close(1)

plt.figure(1)
plt.subplot(211)
plt.plot(v, giv[:,0,0].real, label='SIE')
plt.plot(v, giv_legendre[:,0,0].real, label='Legendre')
plt.xlim([-20, 20])
plt.ylabel(r"Re$G(\mathrm{i}\nu)$")
plt.subplot(212)
plt.plot(v, giv[:,0,0].imag, label='SIE')
plt.plot(v, giv_legendre[:,0,0].imag, marker='x', ls='', label='Legendre')
#plt.plot(v, (1/iv).imag, marker='x', ls='', label='1/iv')
plt.xlim([-20, 20])
#plt.ylim([-0.2,0.2])
plt.xlabel(r"$\nu$")
plt.ylabel(r"Im$G(\mathrm{i}\nu)$")
plt.legend()
plt.savefig("giv.pdf")
plt.close(1)

plt.figure(1)
plt.subplot(211)
plt.plot(v, np.abs(sigma_iv[:,0,0].real), label='SIE')
plt.plot(v, np.abs(sigma_iv_legendre[:,0,0].real), marker='x', ls='', label='legenre')
plt.ylabel(r"|Re$\Sigma(\mathrm{i}\nu)$|")
plt.xscale("log")
plt.yscale("log")
plt.subplot(212)
plt.plot(v, np.abs(sigma_iv[:,0,0].imag), label='SIE')
plt.plot(v, np.abs(sigma_iv_legendre[:,0,0].imag), marker='x', ls='', label='legenre')
plt.xlabel(r"$\nu$")
plt.ylabel(r"|Im$\Sigma(\mathrm{i}\nu)$|")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("sigma_iv.pdf")
plt.close(1)
