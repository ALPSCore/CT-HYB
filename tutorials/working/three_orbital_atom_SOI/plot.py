import numpy as np
import matplotlib.pyplot as plt
from irbasis_x import atom
from irbasis_x.freq import box, to_ph_convention
from alpscthyb.post_proc import QMCResult, VertexEvaluatorAtomED
from alpscthyb import mpi


def plot_comparison(qmc, ref, name, label1='QMC', label2='ref'):
    if mpi.rank != 0:
        return
    qmc = np.moveaxis(qmc, 0, -1).ravel()
    if ref is not None:
        ref = np.moveaxis(ref, 0, -1).ravel()
    fig, axes = plt.subplots(3, 1, figsize=(5,10))
    #axes[0].semilogy(np.abs(qmc.ravel().real), marker='+', ls='', label=label1)
    axes[0].plot(qmc.ravel().real, marker='+', ls='', label=label1)
    if ref is not None:
        #axes[0].semilogy(np.abs(ref.ravel().real), marker='x', ls='', label=label2)
        axes[0].plot(ref.ravel().real, marker='x', ls='', label=label2)
    #axes[1].semilogy(np.abs(qmc.ravel().imag), marker='+', ls='', label=label1)
    axes[1].plot(qmc.ravel().imag, marker='+', ls='', label=label1)
    if ref is not None:
        #axes[1].semilogy(np.abs(ref.ravel().imag), marker='x', ls='', label=label2)
        axes[1].plot(ref.ravel().imag, marker='x', ls='', label=label2)
    axes[0].set_ylabel(r"Re")
    axes[1].set_ylabel(r"Im")

    #axes[2].semilogy(np.abs(qmc), marker='+', ls='', label=label1)
    if ref is not None:
        #axes[2].plot(ref, marker='x', ls='', label=label2)
        axes[2].plot((ref-qmc).real, marker='', ls='--', label='Re diff')
        axes[2].plot((ref-qmc).imag, marker='', ls='-', label='Im diff')
    axes[2].set_ylabel(r"Diff")

    for ax in axes:
        ax.set_xlim([0,250])
        ax.legend()
    fig.tight_layout()
    fig.savefig(name+'.eps')


res = QMCResult('input', verbose=True)
beta = res.beta

ed = VertexEvaluatorAtomED(res.nflavors, res.beta, res.hopping, res.get_asymU())

# Fermionic sampling frequencies
vsample = res.basis_f.wsample
wfs = res.basis_f.wsample
wbs = res.basis_b.wsample
wsample_ffff = box(4, 3, return_conv='full', ravel=True)
wsample_ph = to_ph_convention(*wsample_ffff)

# Fermion-boson frequency box
def box_fb(nf, nb):
    wf = 2*np.arange(-nf,nf)+1
    wb = 2*np.arange(-nb,nb)
    v, w = np.broadcast_arrays(wf[:,None], wb[None,:])
    return v.ravel(), w.ravel()
wsample_fb = box_fb(8, 9)

#SIE
gir_SIE = res.compute_gir_SIE()
giv = res.compute_giv_SIE(vsample)
sigma_iv = res.compute_sigma_iv(giv, vsample)

#Legendre
giv_legendre = res.compute_giv_from_legendre(vsample)
sigma_iv_legendre = res.compute_sigma_iv(giv_legendre, vsample)

# Mixed
giv_mix = res.compute_giv(vsample)
sigma_iv_mix = res.compute_sigma_iv(giv_mix, vsample)

# v_{ab}
print("v_ab: ", res.compute_v())

# G(iv)
nflavors = res.nflavors

# ED data
giv_ref = ed.compute_giv(vsample)
g0iv_ref = ed.compute_g0iv(vsample)

sigma_ref = np.zeros_like(giv_ref)
for i in range(sigma_ref.shape[0]):
    sigma_ref[i,:,:] = \
        np.linalg.inv(g0iv_ref[i,:,:]) - np.linalg.inv(giv_ref[i,:,:])

plot_comparison(
    sigma_iv,
    sigma_ref,
    "sigma_ed", label1='SIE', label2='ED')

plot_comparison(
    giv_mix,
    giv_ref,
    "giv_mix", label1='Mix', label2='ED')

plot_comparison(
    giv_legendre,
    giv_ref,
    "giv_legendre", label1='Legendre', label2='ED')

plot_comparison(
    giv,
    giv_ref,
    "giv_SIE", label1='SIE', label2='ED')
