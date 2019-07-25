import numpy as np
import matplotlib.pyplot as plt
import pylab
import h5py

def read_param(h5, name):
    if '/parameters/dictionary/'+name in h5:
        return h5['/parameters/dictionary/'+name].value
    elif '/parameters/'+name in h5:
        return h5['/parameters/'+name].value
    else:
        raise RuntimeError("Parameter "+ name + " not found") 

def compute_Tnl(n_matsubara, n_legendre):
    Tnl = np.zeros((n_matsubara, n_legendre), dtype=complex)
    for n in xrange(n_matsubara):
        sph_jn = np.array([scipy.special.spherical_jn(l, (n+0.5)*np.pi) for l in range(n_legendre)])
        for il in xrange(n_legendre):
            Tnl[n,il] = ((-1)**n) * ((1J)**(il+1)) * np.sqrt(2*il + 1.0) * sph_jn[il]
    return Tnl

def read_h5(p):
    r = {}

    print p+'.out.h5'
    h5 = h5py.File(p+'.out.h5','r')
    
    r["SITES"] = read_param(h5, 'model.sites')
    r["BETA"] = read_param(h5, 'model.beta')

    def load_g(path):
        N = h5[path].shape[0]
        M = h5[path].shape[1]
        data = h5[path].value.reshape(N,M,M,2)
        return data[:,:,:,0] + 1J*data[:,:,:,1]

    r["Gtau"] = load_g('/gtau/data')

    r["Gomega"] = load_g('/gf/data')

    r["Sign"] = h5['/simulation/results/Sign/mean/value'].value

    r["Sign_count"] = h5['/simulation/results/Sign/count'].value

    return r

# input: G2 in the mix basis
# return G2 in the Matsubara freq. domain: (i,j,k,l, fermionic freq, fermionic freq, bosnic freq)
def compute_G2_matsubara(g2_l, niw_f):
    nl_G2 = g2_l.shape[4]
    Tnl_G2 = compute_Tnl(niw_f, nl_G2)
    tmp = np.tensordot(g2_l, Tnl_G2.conjugate(), axes=(5,1))
    return np.tensordot(Tnl_G2, tmp, axes=(1,4)).transpose((1,2,3,4,0,6,5))

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
    gtau = result_list[i]["Gtau"]
    giw = result_list[i]["Gomega"]

    print "The number of measurements is ", result_list[i]["Sign_count"]

    tau_point = np.linspace(0.0, 1.0, gtau.shape[0])
    plt.subplot(211)
    for i_f in range(nf):
        plt.plot(tau_point, -gtau[:,i_f,i_f].real, color=color_list[i_f], marker='', label='flavor'+str(i_f), ls='--', markersize=0)
        print "flavor ", i_f, "G(tau=0) = ", gtau[0,i_f,i_f].real , "G(tau=beta) = ", gtau[-1,i_f,i_f].real

    omega_point = np.array([(2*im+1)*np.pi/beta for im in xrange(giw.shape[0])])
    plt.subplot(212)
    for i_f in range(nf):
        plt.plot(omega_point, -giw[:,i_f,i_f].imag, color=color_list[i_f], marker='', label='flavor'+str(i_f), ls='--', markersize=0)
    plt.plot(omega_point, 1/omega_point, color='k', label=r'$1/\omega_n$', ls='-')


plt.subplot(211)
plt.legend(loc='best',shadow=True,frameon=False,prop={'size' : 12})

plt.subplot(212)
plt.legend(loc='best',shadow=True,frameon=False,prop={'size' : 12})

plt.tight_layout()
plt.savefig("GF.eps")
plt.close(1)
