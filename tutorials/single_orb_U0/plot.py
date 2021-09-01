import numpy as np
import matplotlib.pyplot as plt
from irbasis_x.freq import box
from alpscthyb.post_proc import QMCResult, VertexEvaluatorU0
from alpscthyb.non_interacting import NoninteractingLimit
from alpscthyb import mpi


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
beta = res.beta

evalU0 = VertexEvaluatorU0(
    res.nflavors, res.beta, res.basis_f, res.basis_b, res.hopping, res.Delta_l)


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

U = 8.
mu = 0.5*U

# Fermionic sampling frequencies
vsample = res.basis_f.wsample
wfs = res.basis_f.wsample
wbs = res.basis_b.wsample

# Fermion-boson frequency box
def box_fb(nf, nb):
    wf = 2*np.arange(-nf,nf)+1
    wb = 2*np.arange(-nb,nb)
    v, w = np.broadcast_arrays(wf[:,None], wb[None,:])
    return v.ravel(), w.ravel()
wsample_fb = box_fb(4, 5)

#SIE
gir_SIE = res.compute_gir_SIE()
giv = res.compute_giv_SIE(vsample)
sigma_iv = res.compute_sigma_iv(giv, vsample)

#Legendre
giv_legendre = res.compute_giv_from_legendre(vsample)
sigma_iv_legendre = res.compute_sigma_iv(giv_legendre, vsample)

# G0
g0iv = evalU0.compute_giv(res.basis_f.wsample)

# Sigma
plot_comparison(
    sigma_iv,
    sigma_iv_legendre,
    "sigma", label1='SIE', label2='Legendre')

# G(iv)
plot_comparison(
    giv,
    giv_legendre,
    "giv", label1='SIE', label2='Legendre')

# vartheta
plot_comparison(
    res.compute_vartheta(wfs),
    evalU0.compute_vartheta(wfs),
    "vartheta")

# varphi & lambda
for name in ['varphi', 'lambda']:
    qmc = getattr(res, f'compute_{name}')(wbs)
    ref = getattr(evalU0, f'compute_{name}')(wbs)
    plot_comparison(qmc, ref, name)

# eta
plot_comparison(res.compute_eta(*wsample_fb), evalU0.compute_eta(*wsample_fb), "eta")

# gamma
plot_comparison(res.compute_gamma(*wsample_fb), evalU0.compute_gamma(*wsample_fb), "gamma")

# h
wsample_ffff = box(4, 3, return_conv='full', ravel=True)
plot_comparison(res.compute_h(wsample_ffff), evalU0.compute_h(wsample_ffff), "h")

# F
F_qmc = res.compute_F(wsample_ffff)
plot_comparison(F_qmc, np.zeros_like(F_qmc), "F")
