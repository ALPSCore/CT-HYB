import numpy as np
import matplotlib.pyplot as plt
from irbasis_x.freq import box
from alpscthyb.post_proc import QMCResult, VertexEvaluatorU0, legendre_to_matsubara, legendre_to_tau
from alpscthyb.non_interacting import NoninteractingLimit


def plot_comparison(qmc, ref, name, label1='QMC', label2='ref'):
    qmc = np.moveaxis(qmc, 0, -1).ravel()
    ref = np.moveaxis(ref, 0, -1).ravel()
    fig, axes = plt.subplots(3, 1, figsize=(5,10))
    #amax = 1.5*np.abs(ref).max()
    axes[0].plot(qmc.ravel().real, marker='+', ls='', label=label1)
    axes[0].plot(ref.ravel().real, marker='x', ls='', label=label2)
    axes[1].plot(qmc.ravel().imag, marker='+', ls='', label=label1)
    axes[1].plot(ref.ravel().imag, marker='x', ls='', label=label2)
    axes[0].set_ylabel(r"Re")
    axes[1].set_ylabel(r"Im")

    axes[2].semilogy(np.abs(qmc), marker='+', ls='', label=label1)
    axes[2].semilogy(np.abs(ref), marker='x', ls='', label=label2)
    axes[2].semilogy(np.abs(ref-qmc), marker='', ls='--', label='diff')
    axes[2].set_ylabel(r"Abs")

    for ax in axes:
        #ax.set_xlim([0,1000])
        ax.legend()
    fig.tight_layout()
    fig.savefig(name+'.eps')


res = QMCResult('input', verbose=True)
non_int = NoninteractingLimit(res)
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

ref_generators = {'lambda': 'lambda_tau', 'varphi': 'varphi_tau'}
for name in ['lambda', 'varphi']:
    data_l = res.__getattribute__(name+'_legendre')
    wsample = 2 * np.arange(-10,10)
    data_iw = legendre_to_matsubara(data_l, wsample)
    tau = np.linspace(0.0, beta, 1000)
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
        plt.semilogy(np.abs(data_l[:,flavors[0],flavors[1],flavors[2],flavors[3]].real), label='Re', marker='x')
        plt.semilogy(np.abs(data_l[:,flavors[0],flavors[1],flavors[2],flavors[3]].imag), label='Im', marker='x')
        plt.legend()
        plt.savefig(name + f"_l_flavors{flavors[0]}{flavors[1]}{flavors[2]}{flavors[3]}.eps")
        plt.close(1)

        plt.figure(1)
        plt.plot(tau, (data_tau[:,flavors[0],flavors[1],flavors[2],flavors[3]].real), label='Re', ls='', marker='x', color='r')
        plt.plot(tau, (data_tau[:,flavors[0],flavors[1],flavors[2],flavors[3]].imag), label='Im', ls='', marker='x', color='b')
        if name in ref_generators:
            data_tau_ref = non_int.__getattribute__(ref_generators[name])(tau)
            plt.plot(tau, (data_tau_ref[:,flavors[0],flavors[1],flavors[2],flavors[3]].real), ls='--', marker='', color='g')
            plt.plot(tau, (data_tau_ref[:,flavors[0],flavors[1],flavors[2],flavors[3]].imag), ls='--', marker='', color='k')
        plt.legend()
        plt.savefig(name + f"_tau_flavors{flavors[0]}{flavors[1]}{flavors[2]}{flavors[3]}.eps")
        plt.close(1)

for name in ['lambda']:
    qmc = getattr(res, f'compute_{name}')(wbs)
    ref = getattr(evalU0, f'compute_{name}')(wbs)
    plot_comparison(qmc[:,0,0,0,0], ref[:,0,0,0,0], name)
