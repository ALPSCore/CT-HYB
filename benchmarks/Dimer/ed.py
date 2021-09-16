import numpy as np
from alpscthyb.post_proc import QMCResult, VertexEvaluatorED
from alpscthyb.exact_diag import *
from alpscthyb.occupation_basis import *
from alpscthyb.interaction import slater_kanamori_asymm
from alpscthyb import mpi
from matplotlib import pylab as plt
from irbasis_x.freq import box

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
    ymin = 1e-4*np.abs(ref).max()
    if ref is not None:
        axes[2].semilogy(np.abs(ref), marker='x', ls='', label=label2)
        axes[2].semilogy(np.abs(ref-qmc), marker='+', ls='--', label='diff')
    axes[2].set_ylabel(r"Abs")
    axes[2].set_ylim([ymin,None])

    for ax in axes:
        #ax.set_xlim([400,600])
        #ax.set_ylim([-0.5,0.5])
        #ax.set_ylim([-1,1])
        ax.legend()
    fig.tight_layout()
    fig.savefig(name+'.eps')

# QMC data
res = QMCResult('input', verbose=True)
beta = res.beta
norb = res.nflavors//2

# ED
nflavors = 8

eps_bath = np.array([0.27, -0.4])  # Bath site energies
t_bath = 0.0    

hopping_bath = np.zeros(2*(2,2), dtype=np.complex128)
hopping_coup = np.zeros(2*(2,2), dtype=np.complex128)
for sp in range(2):
    hopping_bath[:,sp,:,sp] = np.diag(eps_bath) - np.matrix([[0, t_bath], [t_bath, 0]])
    hopping_coup[:,sp,:,sp] = np.ones((2,2))
hopping_bath = hopping_bath.reshape((4, 4))
hopping_coup = hopping_coup.reshape((4, 4))

ed = VertexEvaluatorED(beta, res.hopping, res.get_asymU(), hopping_bath, hopping_coup)

wfs = 2*np.arange(0,100)+1
wbs = 2*np.arange(0,100)


### Self-energy ###
sigma_iv_ed = res.compute_sigma_iv(ed.compute_giv(wfs), wfs)

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


# varphi & lambda
for name in ['varphi', 'lambda']:
    qmc = getattr(res, f'compute_{name}')(wbs)
    ref = getattr(ed,  f'compute_{name}')(wbs)
    plot_comparison(qmc, ref, name)

# eta & gamma
def box_fb(nf, nb):
    wf = 2*np.arange(-nf,nf)+1
    wb = 2*np.arange(-nb,nb)
    v, w = np.broadcast_arrays(wf[:,None], wb[None,:])
    return v.ravel(), w.ravel()
wsample_fb = box_fb(8, 9)

# eta
eta_ref = ed.compute_eta(*wsample_fb)
plot_comparison(res.compute_eta(*wsample_fb), eta_ref, "eta")

# gamma
gamma_ref = ed.compute_gamma(*wsample_fb)
plot_comparison(res.compute_gamma(*wsample_fb), gamma_ref, "gamma")

# h
wsample_ffff = box(4, 3, return_conv='full', ravel=True)
h_ref = ed.compute_h(wsample_ffff)
plot_comparison(res.compute_h(wsample_ffff), h_ref, "h")

# F
F_ref = ed.compute_F(wsample_ffff)
F_qmc = res.compute_F(wsample_ffff)
plot_comparison(F_qmc, F_ref, "F")