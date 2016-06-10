import sys
import numpy as np
from scipy.integrate import simps
from math import pi

#Lambda = 1.0
Lambda = 1.0
nf=6
norb=nf/2
Uval = 1.0
Jval = 0.25*Uval

Himp = -0.5*Lambda*np.array( [
        [ 0,  0, -1J,  0,  0,  1],
        [ 0,  0,  0,  1J, -1,  0],
        [1J,  0,  0,   0,  0,-1J],
        [ 0,-1J,  0,   0,-1J,  0],
        [ 0, -1,  0,  1J,  0,  0],
        [ 1,  0, 1J,   0,  0,  0]], dtype=complex)

mu = 2.5*Uval-5*Jval
for flavor in xrange(nf):
    Himp[flavor,flavor] -= mu

f = open('hopping.txt','w')
for iorb in xrange(nf):
    for jorb in xrange(nf):
        print>>f, iorb, jorb, Himp[iorb,jorb].real, Himp[iorb,jorb].imag
f.close()

