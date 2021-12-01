import numpy as np
import matplotlib.pyplot as plt
from irbasis_x import atom
from irbasis_x.freq import box, to_ph_convention
from alpscthyb.post_proc import QMCResult, VertexEvaluatorAtomED
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
        axes[2].semilogy(np.abs(ref-qmc), marker='', ls='--', label='diff')
    axes[2].set_ylabel(r"Abs")

    for ax in axes:
        ax.set_xlim([000,600])
        ax.legend()
    fig.tight_layout()
    fig.savefig(name+'.eps')


res = QMCResult('input', verbose=True)
beta = res.beta

ed = VertexEvaluatorAtomED(res.nflavors, res.beta, res.hopping, res.get_asymU())

# pyed data
with h5py.File('results/pyed.h5', 'r') as h5:
    giv_ed = float_to_complex_array(h5['/G/bl/data'][()])

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

# Sigma
#plot_comparison(
    #sigma_iv,
    #sigma_iv_legendre,
    #"sigma", label1='SIE', label2='Legendre')
#
## G(iv)
#nflavors = res.nflavors
#
## ED data
#giv_ref = ed.compute_giv(wfs)
#g0iv_ref = ed.compute_g0iv(wfs)
#
#sigma_ref = np.zeros_like(giv_ref)
#for i in range(sigma_ref.shape[0]):
    #sigma_ref[i,:,:] = \
        #np.linalg.inv(g0iv_ref[i,:,:]) - np.linalg.inv(giv_ref[i,:,:])

#plot_comparison(
    #giv,
    #giv_legendre,
    #"giv", label1='SIE', label2='Legendre')
#
#plot_comparison(
    #giv,
    #giv_ref,
    #"giv_ed", label1='SIE', label2='ED')
