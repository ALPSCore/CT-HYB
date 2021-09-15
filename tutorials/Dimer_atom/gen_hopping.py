import sys
import numpy as np
from scipy.integrate import simps
from math import pi

nf=4
norb=nf/2

mu = 0.0                        # Chemical potential
#eps = np.array([0.0, 0.1])         # Impurity site energies
eps = np.array([0.0, 0.0])         # Impurity site energies
#t = 0.2 
t = 0.0


hopping = np.zeros(2*(nf//2,2), dtype=np.complex128)
for sp in range(2):
    hopping[0:2,sp,0:2,sp] = np.diag(eps - mu) - np.matrix([[0, t], [t, 0]])
hopping = hopping.reshape((nf,nf))

hopping_rnd = 1e-5*np.random.randn(nf, nf)
hopping_rnd = hopping_rnd + hopping_rnd.T
hopping += hopping_rnd

f = open('hopping.txt','w')
for iorb in range(nf):
    for jorb in range(nf):
        print(iorb, jorb, hopping[iorb,jorb].real, hopping[iorb,jorb].imag, file=f)
f.close()

