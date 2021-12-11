import numpy as np
import matplotlib.pyplot as plt
from irbasis_x import atom
from irbasis_x.freq import box, to_ph_convention
from alpscthyb.post_proc import QMCResult, VertexEvaluatorAtomED, VertexEvaluatorED
from alpscthyb import mpi

from model import *

def plot_comparison(qmc, ref, name, label1='QMC', label2='ref'):
    if mpi.rank != 0:
        return
    qmc = np.moveaxis(qmc, 0, -1).ravel()
    if ref is not None:
        ref = np.moveaxis(ref, 0, -1).ravel()
    fig, axes = plt.subplots(3, 1, figsize=(5,10))
    #amax = 1.5*np.abs(ref).max()
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
        ax.legend()
    fig.tight_layout()
    fig.savefig(name+'.eps')


res = QMCResult('input', verbose=True)
assert np.abs(beta - res.beta) < 1e-8

ed = VertexEvaluatorED(beta, H0, asymU, Hbath, V)

#plt.figure(1)
#for f in range(res.nflavors):
    #plt.plot(res.Delta_tau[:,f,f].real, label=f'flavor{f}')
    ##plt.plot(res.Delta_tau_rec[:,f,f].real, label=f'flavor{f}')
#plt.legend()
#plt.savefig("Delta_tau.eps")
#plt.close(1)
#
#plt.figure(1)
#for f in range(res.nflavors):
    #plt.semilogy(np.abs(res.Delta_l[:,f,f]), label=f'flavor{f}')
#plt.legend()
#plt.savefig("Delta_l.eps")
#plt.close(1)

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

# v_{ab}
print("v_ab: ", res.compute_v())

# Sigma
plot_comparison(
    sigma_iv,
    sigma_iv_legendre,
    "sigma", label1='SIE', label2='Legendre')

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
    "sigma_SIE", label1='SIE', label2='ED')

plot_comparison(
    giv,
    giv_legendre,
    "giv", label1='SIE', label2='Legendre')

plot_comparison(
    giv,
    giv_ref,
    "giv_ed", label1='SIE', label2='ED')

lambda_ref = ed.compute_lambda(wbs)
vartheta_ref = ed.compute_vartheta(wfs)
eta_ref = ed.compute_eta(*wsample_fb)
gamma_ref = ed.compute_gamma(*wsample_fb)
h_ref = ed.compute_h(wsample_ffff)
F_ref = ed.compute_F(wsample_ffff)

# vartheta
plot_comparison(
    res.compute_vartheta(wfs),
    vartheta_ref,
    "vartheta")

# varphi & lambda
for name in ['varphi', 'lambda']:
    qmc = getattr(res, f'compute_{name}')(wbs)
    ref = getattr(ed,  f'compute_{name}')(wbs)
    plot_comparison(qmc, ref, name)

# eta
plot_comparison(res.compute_eta(*wsample_fb), eta_ref, "eta")

# gamma
plot_comparison(res.compute_gamma(*wsample_fb), gamma_ref, "gamma")

# h
plot_comparison(res.compute_h(wsample_ffff), h_ref, "h")

# F
#v1 = np.array([1, 11, 101, 1001, 10001])
#v2 = np.array([100001, 100001, 100001, 100001, 100001])
#v3 = np.array([200001, 200001, 200001, 200001, 200001])
#v4 = v1 - v2 + v3
#wsample_ffff = (v1, v2, v3, v4)
#wsample_ph = to_ph_convention(*wsample_ffff)
#F_ref = beta * _atomic_F_ph(U, beta, wsample_ph)
F_qmc = res.compute_F(wsample_ffff)
plot_comparison(F_qmc, F_ref, "F")

if mpi.rank == 0:
    for idx_w in range(F_qmc.shape[0]):
        print(wsample_ffff[0][idx_w],
              wsample_ffff[1][idx_w],
              wsample_ffff[2][idx_w],
              wsample_ffff[3][idx_w],
              F_qmc[idx_w,0,0,1,1].real,
              F_qmc[idx_w,0,0,1,1].imag,
              F_ref[idx_w,0,0,1,1].real,
              F_ref[idx_w,0,0,1,1].imag
          )
              #np.sum(np.abs(F[idx_w,...])),
              #np.sum(np.abs(F_ref[idx_w,...]))
