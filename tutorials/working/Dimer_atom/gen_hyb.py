import numpy as np
from scipy.integrate import simps
from math import pi

ndiv_tau = 10000
nf = 4

f = open('delta.txt', 'w')
for i in range(ndiv_tau+1):
    for j in range(nf):
        for k in range(nf):
            if j==k:
                print(i, j, k, 1e-8, 0.0, file=f)
            else:
                print(i, j, k, 0.0, 0.0, file=f)
f.close()
