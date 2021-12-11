import numpy as np
import sys
from math import pi
from scipy.integrate import simps
import random

up = 0
down = 1

def generate_U_tensor_onsite(n_orb, U):
    U_tensor = np.zeros((n_orb,2,n_orb,2,n_orb,2,n_orb,2),dtype=complex)

    for iorb in range(n_orb):
        U_tensor[iorb, up, iorb, down, iorb, down, iorb, up] = U
        U_tensor[iorb, down, iorb, up, iorb, up, iorb, down] = U

    return U_tensor, 2*n_orb

def ft_to_tau_hyb(ntau, beta, matsubara_freq, tau, Vek, data_n, data_tau, cutoff):
 for it in range(ntau+1):
     tau_tmp=tau[it]
     if it==0:
         tau_tmp=1E-4*(beta/ntau)
     if it==ntau:
         tau_tmp=beta-1E-4*(beta/ntau)
     ztmp=0.0
     for im in range(cutoff):
         ztmp+=(data_n[im]+1J*Vek/matsubara_freq[im])*np.exp(-1J*matsubara_freq[im]*tau_tmp)
     ztmp=ztmp/beta
     data_tau[it]=2.0*ztmp.real-0.5*Vek


#Parameters
n_site = 1
nf = n_site*2

Uval = 1e-8
vbeta = 10.0
ntau = 1000
mu = 0.5*Uval
V = 1.0e-3
H = 0.0

#Generate Coulomb tensor
U_tensor, num_elem = generate_U_tensor_onsite(n_site, Uval)
f = open("Uijkl.txt", "w")
print(num_elem, file=f)
line = 0
for iorb1 in range(n_site):
    for iorb2 in range(n_site):
        for iorb3 in range(n_site):
            for iorb4 in range(n_site):
                for isp in range(2):
                    for isp2 in range(2):
                        if U_tensor[iorb1,isp,iorb2,isp2,iorb3,isp2,iorb4,isp] != 0.0:
                            print(line, "   ", 2*iorb1+isp, 2*iorb2+isp2, 2*iorb3+isp2, 2*iorb4+isp, U_tensor[iorb1,isp,iorb2,isp2,iorb3,isp2,iorb4,isp].real, U_tensor[iorb1,isp,iorb2,isp2,iorb3,isp2,iorb4,isp].imag, file=f)
                            line += 1

f.close()

#Generate hopping matrix
f = open('hopping.txt','w')
for iorb in range(nf):
    for jorb in range(nf):
        if iorb == jorb:
            print(iorb, jorb, -mu - H*(-1)**iorb, 0.0, file=f)
        else:
            print(iorb, jorb, 0.0, 0.0, file=f)
f.close()


#Generate hybridization function
matsubara_freq=np.zeros((ntau,),dtype=float)
for im in range(ntau):
    matsubara_freq[im]=((2*im+1)*np.pi)/vbeta

tau=np.zeros((ntau+1,),dtype=float)
for it in range(ntau+1):
    tau[it]=(vbeta/ntau)*it

ndiv_dos = 10000
W=2.0
e_smp=np.linspace(-W,W,ndiv_dos)
dos_smp=np.sqrt(W**2-e_smp**2)/(0.5*pi*W**2)
ek_var = simps(dos_smp*(e_smp**2), e_smp)

#Bath Green's function (g_omega, g_tau)
g_omega=np.zeros((ntau,),dtype=complex)
g_tau=np.zeros((ntau+1,),dtype=complex)
for im in range(ntau):
    f_tmp = dos_smp/(1J*matsubara_freq[im]-e_smp)
    g_omega[im]=simps(f_tmp,e_smp)

ft_to_tau_hyb(ntau,vbeta,matsubara_freq,tau,1.0,g_omega,g_tau,ntau)

f = open('delta.txt', 'w')
for i in range(ntau+1):
    for j in range(nf):
        for k in range(nf):
            if j==k:
                print(i, j, k, (V**2)*g_tau[i].real, (V**2)*g_tau[i].imag, file=f)
            else:
                print(i, j, k, 0.0, 0.0, file=f)
f.close()
