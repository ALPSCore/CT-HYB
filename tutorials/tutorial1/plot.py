import numpy as np
import matplotlib.pyplot as plt
import pylab
import h5py

def read_h5(p):
    r = {}

    print p+'.out.h5'
    h5 = h5py.File(p+'.out.h5','r')
    
    r["N_TAU"] = h5['/parameters/N_TAU'].value
    r["SITES"] = h5['/parameters/SITES'].value
    r["BETA"] = h5['/parameters/BETA'].value

    r["Gtau"] = h5['/gtau/data'].value.reshape(r["N_TAU"]+1,r["SITES"]*2,r["SITES"]*2,2)
    r["Gtau"] = r["Gtau"][:,:,:,0] + 1J*r["Gtau"][:,:,:,1]

    r["Gomega"] = h5['/gf/data'].value.reshape(r["N_TAU"],r["SITES"]*2,r["SITES"]*2,2)
    r["Gomega"] = r["Gomega"][:,:,:,0] + 1J*r["Gomega"][:,:,:,1]

    r["Sign"] = h5['/simulation/results/Sign/mean/value'].value
    r["n"] = h5['/simulation/results/n/mean/value'].value

    return r

#prefix_list = ['input', 'input-cutoff']
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
#plt.xlabel(r'$U$ (eV)', fontname='serif')
#plt.ylabel(r'$L_{111}$ ($\mu_B$)', fontname='serif')
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
    ntau = result_list[i]["N_TAU"]
    norb = result_list[i]["SITES"]
    beta = result_list[i]["BETA"]
    nf = norb*2

    xdata = np.array([k/float(ntau) for k in range(ntau+1)])
    sign = result_list[i]["Sign"]
    gf_legendre = result_list[i]["Gtau"]
    gomega_l = result_list[i]["Gomega"]

    print "sign=",sign
    occ = 0.0
    for i_f in range(nf):
        occ += -gf_legendre[-1,i_f,i_f]

    tau_point = np.linspace(0.0, 1.0, ntau+1)
    plt.subplot(211)
    for i_f in range(nf):
        plt.plot(tau_point, -gf_legendre[:,i_f,i_f].real, color=color_list[i_f], marker='', label='flavor'+str(i_f), ls='--', markersize=0)
    #f = open('Gtau_l.txt', 'w')
    #for i_f in range(nf):
        #for i_f2 in range(nf):
            #print >>f, "#flavor ", i_f, i_f2
            #for itau in xrange(ntau+1):
                #print >>f, itau,
                #print >>f, gf_legendre[itau,i_f,i_f2].real, gf_legendre[itau,i_f,i_f2].imag,
                #print >>f, ""
            #print >>f, ""
    #f.close()

    omega_point = np.array([(2*im+1)*np.pi/beta for im in xrange(ntau)])
    plt.subplot(212)
    for i_f in range(nf):
        plt.plot(omega_point, -gomega_l[:,i_f,i_f].imag, color=color_list[i_f], marker='', label='flavor'+str(i_f), ls='--', markersize=0)
    plt.plot(omega_point, 1/omega_point, color='k', label=r'$1/\omega_n$', ls='-')
    #f = open('Gomega_l.txt', 'w')
    #for i_f in range(nf):
        #for i_f2 in range(nf):
            #print >>f, "#flavor ", i_f, i_f2
            #for itau in xrange(ntau):
                #print >>f, itau,
                #print >>f, gomega_l[itau,i_f,i_f2].real, gomega_l[itau,i_f,i_f2].imag,
                #print >>f, ""
            #print >>f, ""
    #f.close()

plt.subplot(211)
plt.legend(loc='best',shadow=True,frameon=False,prop={'size' : 12})

plt.subplot(212)
plt.legend(loc='best',shadow=True,frameon=False,prop={'size' : 12})

plt.tight_layout()
plt.savefig("GF.pdf")
plt.close(1)
