import numpy as np
import matplotlib.pyplot as plt
from alpscthyb.post_proc import QMCResult, VertexEvaluatorAtomED
from alpscthyb.util import *

res = QMCResult('input', verbose=True)
beta = res.beta

ed = VertexEvaluatorAtomED(res.nflavors, res.beta, res.hopping, res.get_asymU())

# Fermion-boson frequency box
def box_fb(nf, nb):
    wf = 2*np.arange(-nf,nf)+1
    wb = 2*np.arange(-nb,nb)
    v, w = np.broadcast_arrays(wf[:,None], wb[None,:])
    return v.ravel(), w.ravel()
wsample_fb = box_fb(8, 3)

#wsample_ffff = box(4, 3, return_conv='full', ravel=True)
#wsample_ph = to_ph_convention(*wsample_ffff)


#SIE
gir_SIE = res.compute_gir_SIE()
giv = res.compute_giv_SIE(wfs)
sigma_iv = res.compute_sigma_iv(giv, wfs)



# v_{ab}
print("v_ab: ", res.compute_v())

# G(iv)
nflavors = res.nflavors

# ED data
giv_ref = ed.compute_giv(wfs)
g0iv_ref = ed.compute_g0iv(wfs)

sigma_ref = np.zeros_like(giv_ref)
for i in range(sigma_ref.shape[0]):
    sigma_ref[i,:,:] = \
        np.linalg.inv(g0iv_ref[i,:,:]) - np.linalg.inv(giv_ref[i,:,:])

plot_comparison(
    sigma_iv,
    sigma_ref,
    "sigma_ed", label1='SIE', label2='ED')

#Legendre
if hasattr(res, "gl"):
    giv_legendre = res.compute_giv_from_legendre(wfs)
    sigma_iv_legendre = res.compute_sigma_iv(giv_legendre, wfs)
    plot_comparison(
        giv_legendre,
        giv_ref,
        "giv_legendre", label1='Legendre', label2='ED')

if hasattr(res, "gIR"):
    giv_IR = res.compute_giv_from_IR(wfs)
    sigma_iv_IR = res.compute_sigma_iv(giv_IR, wfs)
    plot_comparison(
        giv_IR,
        giv_ref,
        "giv_IR", label1='IR', label2='ED')

plot_comparison(
    giv,
    giv_ref,
    "giv_SIE", label1='SIE', label2='ED')
