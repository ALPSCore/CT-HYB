import sys
import numpy as np
from scipy.integrate import simps
from math import pi

nf=4
norb=nf/2

f = open('hopping.txt','w')
for iorb in xrange(nf):
    for jorb in xrange(nf):
        print>>f, iorb, jorb, 0.0, 0.0
f.close()

