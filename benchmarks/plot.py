import numpy as np
import matplotlib.pyplot as plt
from alpscthyb.post_proc import QMCResult, reconst_vartheta
from alpscthyb.util import float_to_complex_array
from alpscthyb import mpi
import h5py



def plot_comparison(qmc, ref, name, label1='QMC', label2='ref'):
    if mpi.rank != 0:
        return
    qmc = np.moveaxis(qmc, 0, -1).ravel()
    if ref is not None:
        ref = np.moveaxis(ref, 0, -1).ravel()
    fig, axes = plt.subplots(3, 1, figsize=(5,10))
    axes[0].plot(qmc.ravel().real, marker='+', ls='', label=label1)
    if ref is not None:
        axes[0].plot(ref.ravel().real, marker='x', ls='', label=label2)
    axes[1].plot(qmc.ravel().imag, marker='+', ls='', label=label1)
    if ref is not None:
        axes[1].plot(ref.ravel().imag, marker='x', ls='', label=label2)
    axes[0].set_ylabel(r"Re")
    axes[1].set_ylabel(r"Im")

    axes[2].semilogy(np.abs(qmc), marker='+', ls='', label=label1)
    if ref is not None:
        axes[2].semilogy(np.abs(ref), marker='x', ls='', label=label2)
        axes[2].semilogy(np.abs(ref-qmc), marker='+', ls='--', label='diff')
    axes[2].set_ylabel(r"Abs")

    for ax in axes:
        ax.set_xlim([40,60])
        ax.set_ylim([-0.5,0.5])
        #ax.set_ylim([-1,1])
        ax.legend()
    fig.tight_layout()
    fig.savefig(name+'.eps')


res = QMCResult('input', verbose=True)
beta = res.beta
norb = res.nflavors//2

#ed = VertexEvaluatorAtomED(res.nflavors, res.beta, res.hopping, res.get_asymU())

# ED data
with h5py.File('results/pyed.h5', 'r') as h5:
    nsp = 2
    gup = float_to_complex_array(h5['/G/up/data'][()])
    gdn = float_to_complex_array(h5['/G/dn/data'][()])
    giv_ed = np.zeros((gup.shape[0], norb, 2, norb, 2), dtype=np.complex128)
    giv_ed[:, :, 0, :, 0] = gup
    giv_ed[:, :, 1, :, 1] = gdn
    giv_ed = giv_ed.reshape((gup.shape[0], 2*norb, 2*norb))

    dens_mat_ed = np.zeros((norb, 2, norb, 2), dtype=np.complex128)
    dens_mat_ed[:, 0, :, 0] = float_to_complex_array(h5['/dens_mat/up'][()])
    dens_mat_ed[:, 1, :, 1] = float_to_complex_array(h5['/dens_mat/dn'][()])
    dens_mat_ed = dens_mat_ed.reshape((2*norb, 2*norb))

print("Density matrix")
print("QMC: ", res.get_dm())
print("ED: ", dens_mat_ed)
print("diff: ", res.get_dm()-dens_mat_ed)

# Fermionic sampling frequencies
wfs = 2*np.arange(-giv_ed.shape[0]//2, giv_ed.shape[0]//2) + 1
sigma_iv_ed = res.compute_sigma_iv(giv_ed, wfs)

#SIE
gir_SIE = res.compute_gir_SIE()
giv = res.compute_giv_SIE(wfs)
sigma_iv = res.compute_sigma_iv(giv, wfs)

#Legendre
giv_legendre = res.compute_giv_from_legendre(wfs)
sigma_iv_legendre = res.compute_sigma_iv(giv_legendre, wfs)

plt.figure(1)
for f in range(res.nflavors):
    plt.plot(res.Delta_tau[:,f,f].real, label=f'flavor{f}')
    plt.plot(res.Delta_tau_rec[:,f,f].real, label=f'flavor{f}')
plt.legend()
plt.savefig("Delta_tau.eps")
plt.close(1)

plt.figure(1)
for f in range(res.nflavors):
    plt.semilogy(np.abs(res.Delta_l[:,f,f]), label=f'flavor{f}')
plt.legend()
plt.savefig("Delta_l.eps")
plt.close(1)

plot_comparison(
    sigma_iv,
    sigma_iv_ed,
    "sigma_SIE", label1='SIE', label2='ED')

plot_comparison(
    sigma_iv_legendre,
    sigma_iv_ed,
    "sigma_legendre", label1='Legendre', label2='ED')

plot_comparison(
    giv_legendre,
    giv_ed,
    "giv_legendre", label1='Legendre', label2='ED')

plot_comparison(
    giv,
    giv_ed,
    "giv_SIE", label1='SIE', label2='ED')

vartheta_ed = reconst_vartheta(res.get_asymU(), giv_ed, res.compute_g0iv(wfs), dens_mat_ed)
vartheta_legendre = reconst_vartheta(res.get_asymU(), giv_legendre, res.compute_g0iv(wfs), dens_mat_ed)

plot_comparison(
    res.compute_vartheta(wfs),
    vartheta_ed,
    "vartheta_SIE", label1='SIE', label2='ED')

plot_comparison(
    vartheta_legendre,
    vartheta_ed,
    "vartheta_legendre", label1='legendre', label2='ED')
