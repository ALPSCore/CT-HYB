import numpy as np
from model import *
from itertools import product
from irbasis_x.twopoint import FiniteTemperatureBasis, TruncatedBasis
import irbasis

ntau = 1000
Lambda = 1E+5

nflavors = H0.shape[0]

with open("Uijkl.txt", "w") as f:
    print(nflavors**4, file=f)
    line = 0
    for i, j, k, l in product(range(nflavors), repeat=4):
        print(line, "   ", i, j, l, k, 0.5*asymU[i,k,j,l].real, 0.5*asymU[i,k,j,l].imag, file=f)
        line += 1

#Generate hopping matrix
with open('hopping.txt','w') as f:
    for iorb in range(nflavors):
        for jorb in range(nflavors):
            print(iorb, jorb, H0[iorb,jorb].real, H0[iorb,jorb].imag, file=f)

basis_f = FiniteTemperatureBasis(TruncatedBasis(irbasis.load('F', Lambda)), beta)

tau = np.linspace(0, beta, ntau+1)

prj_tau = basis_f.Ultau_all_l(tau).T
prj_iv = basis_f.Uwl(basis_f.wsample)

Delta_iv = ed.compute_Delta_iv(basis_f.wsample)

Delta_l = np.einsum('lv,vij->lij', np.linalg.pinv(prj_iv), Delta_iv)
Delta_tau = np.einsum('tl,lij->tij', prj_tau, Delta_l)

#Generate hybridization function
with open('delta.txt', 'w') as f:
    for i in range(ntau+1):
        for j in range(nflavors):
            for k in range(nflavors):
                print(i, j, k, Delta_tau[i,j,k].real, Delta_tau[i,j,k].imag, file=f)

# input file
input_str = f"""\
seed=10
timelimit=600
verbose=1

model.sites=1
model.spins=2
model.flavors=2
# Or, you can use sites=2, spins=1.
# Only the difference is that a global spin flip is not attempted for this alternative choice.
model.coulomb_tensor_input_file='Uijkl.txt'
model.hopping_matrix_input_file='hopping.txt'
model.beta={beta}

#Delta(tau)
model.delta_input_file='delta.txt'

#The number of tau points in Delta(tau) - 1
model.n_tau_hyb={ntau}

measurement.G1.n_legendre=50
measurement.G1.n_tau=1000
measurement.G1.n_matsubara=500

measurement.G2.SIE.on = 1 
"""

with open('input.ini', 'w') as f:
    print(input_str, file=f)
