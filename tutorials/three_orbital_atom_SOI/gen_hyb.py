import numpy as np
from scipy.integrate import simps
from math import pi

def ft_to_tau_hyb(ndiv_tau, beta, matsubara_freq, tau, Vek, data_n, data_tau, cutoff):
 for it in range(ndiv_tau+1):
     tau_tmp=tau[it]
     if it==0:
         tau_tmp=1E-4*(beta/ndiv_tau)
     if it==ndiv_tau:
         tau_tmp=beta-1E-4*(beta/ndiv_tau)
     ztmp=0.0
     for im in range(cutoff):
         ztmp+=(data_n[im]+1J*Vek/matsubara_freq[im])*np.exp(-1J*matsubara_freq[im]*tau_tmp)
     ztmp=ztmp/beta
     data_tau[it]=2.0*ztmp.real-0.5*Vek


vbeta=20.0
ndiv_tau=1000
nf=6
V = 1e-5

matsubara_freq=np.zeros((ndiv_tau,),dtype=float)
for im in range(ndiv_tau):
    matsubara_freq[im]=((2*im+1)*np.pi)/vbeta

tau=np.zeros((ndiv_tau+1,),dtype=float)
for it in range(ndiv_tau+1):
    tau[it]=(vbeta/ndiv_tau)*it

ndiv_dos = 10000
W=2.0
e_smp=np.linspace(-W,W,ndiv_dos)
dos_smp=np.sqrt(W**2-e_smp**2)/(0.5*pi*W**2)
ek_var = simps(dos_smp*(e_smp**2), e_smp)

#Bath Green's function (g_omega, g_tau)
g_omega=np.zeros((ndiv_tau,),dtype=complex)
g_tau=np.zeros((ndiv_tau+1,),dtype=complex)
for im in range(ndiv_tau):
    f_tmp = dos_smp/(1J*matsubara_freq[im]-e_smp)
    g_omega[im]=simps(f_tmp,e_smp)

ft_to_tau_hyb(ndiv_tau,vbeta,matsubara_freq,tau,1.0,g_omega,g_tau,ndiv_tau)

f = open('delta.txt', 'w')
for i in range(ndiv_tau+1):
    for j in range(nf):
        for k in range(nf):
            if j==k:
                print(i, j, k, (V**2)*g_tau[i].real, (V**2)*g_tau[i].imag, file=f)
            else:
                print(i, j, k, 0.0, 0.0, file=f)
f.close()
