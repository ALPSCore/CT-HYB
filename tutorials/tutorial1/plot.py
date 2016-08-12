import numpy as np
import matplotlib.pyplot as plt
import pylab
import h5py

def read_h5(p):
    r = {}

    print p+'.out.h5'
    h5 = h5py.File(p+'.out.h5','r')
    
    r["SITES"] = h5['/parameters/model.sites'].value
    r["BETA"] = h5['/parameters/model.beta'].value

    def load_g(path):
        N = h5[path].shape[0]
        M = h5[path].shape[1]
        data = h5[path].value.reshape(N,M,M,2)
        return data[:,:,:,0] + 1J*data[:,:,:,1]

    r["Gtau"] = load_g('/gtau/data')

    r["Gomega"] = load_g('/gf/data')

    r["Sign"] = h5['/simulation/results/Sign/mean/value'].value

    return r

prefix_list = ['input']
result_list = []
for p in prefix_list:
    result_list.append(read_h5(p))

color_list = ['r', 'g', 'b', 'y', 'k', 'm']
params = {
    'backend': 'ps',
    'axes.labelsize': 24,
    'text.fontsize': 24,
    'legend.fontsize': 18,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'text.usetex': True,
    }
pylab.rcParams.update(params)
plt.figure(1,figsize=(8,8))
plt.subplot(211)
plt.xlabel(r'$\tau/\beta$', fontname='serif')
plt.ylabel(r'$-\mathrm{Re}G(\tau)$', fontname='serif')
plt.yscale('log')

plt.subplot(212)
plt.xlabel(r'$\omega_n$', fontname='serif')
plt.ylabel(r'$-\mathrm{Im}G(i\omega_n)$', fontname='serif')
plt.xscale('log')
plt.yscale('log')

for i in range(len(result_list)):
    norb = result_list[i]["SITES"]
    beta = result_list[i]["BETA"]
    nf = norb*2

    sign = result_list[i]["Sign"]
    gf_legendre = result_list[i]["Gtau"]
    gomega_l = result_list[i]["Gomega"]

    print "sign=",sign
    occ = 0.0
    for i_f in range(nf):
        occ += -gf_legendre[-1,i_f,i_f]

    tau_point = np.linspace(0.0, 1.0, gf_legendre.shape[0])
    plt.subplot(211)
    for i_f in range(nf):
        plt.plot(tau_point, -gf_legendre[:,i_f,i_f].real, color=color_list[i_f], marker='', label='flavor'+str(i_f), ls='--', markersize=0)

    omega_point = np.array([(2*im+1)*np.pi/beta for im in xrange(gomega_l.shape[0])])
    plt.subplot(212)
    for i_f in range(nf):
        plt.plot(omega_point, -gomega_l[:,i_f,i_f].imag, color=color_list[i_f], marker='', label='flavor'+str(i_f), ls='--', markersize=0)
    plt.plot(omega_point, 1/omega_point, color='k', label=r'$1/\omega_n$', ls='-')

plt.subplot(211)
plt.legend(loc='best',shadow=True,frameon=False,prop={'size' : 12})

plt.subplot(212)
plt.legend(loc='best',shadow=True,frameon=False,prop={'size' : 12})

plt.tight_layout()
plt.savefig("GF.pdf")
plt.close(1)
