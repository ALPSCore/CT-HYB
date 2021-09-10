import numpy as np
from alpscthyb.post_proc import QMCResult
from alpscthyb.exact_diag import *
from alpscthyb.interaction import slater_kanamori_asymm

nflavors = 8

beta = 5.                       # Inverse temperature
mu = 0.0                        # Chemical potential
eps = np.array([0.0, 0.1])         # Impurity site energies
t = 0.2 

eps_bath = np.array([0.27, -0.4])  # Bath site energies
t_bath = 0.0    

U = 2.                          # On-site interaction
V = 1.                          # Intersite interaction
J = 0.5                         # Hunds coupling

hopping = np.zeros(2*(nflavors//2,2), dtype=np.complex128)
for sp in range(2):
    hopping[0:2,sp,0:2,sp] = np.diag(eps - mu) - np.matrix([[0, t], [t, 0]])
    hopping[0:2,sp,2:4,sp] = np.identity(2)
    hopping[2:4,sp,0:2,sp] = np.identity(2)
    hopping[2:4,sp,2:4,sp] = np.diag(eps_bath) - np.matrix([[0, t_bath], [t_bath, 0]])
hopping_atom = hopping[0:2,:,0:2,:].reshape((4,4))
hopping = hopping.reshape((nflavors,nflavors))

asymU_atom = slater_kanamori_asymm(2, U, J)
asymU_atom = check_asymm(asymU_atom)
asymU = np.zeros(4*(nflavors,), dtype=np.complex128)
asymU[0:4,0:4,0:4,0:4] = asymU_atom
asymU = check_asymm(asymU)

# Local hamiltonian
_, cdag_ops_atom = construct_cdagger_ops(4)
ham_atom = construct_ham(hopping_atom, asymU_atom, cdag_ops_atom)
evals_atom, evecs_atom = np.linalg.eigh(ham_atom.toarray())

# Hamiltonian of whole system
_, cdag_ops = construct_cdagger_ops(nflavors)
ham = construct_ham(hopping, asymU, cdag_ops)
evals, evecs = np.linalg.eigh(ham.toarray())
print(evals)
