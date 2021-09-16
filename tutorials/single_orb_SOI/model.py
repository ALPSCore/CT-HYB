import numpy as np
from alpscthyb.post_proc import VertexEvaluatorED
from alpscthyb.interaction import *

# H0
H0 = np.zeros((2,2), dtype=np.complex128)
t_imp = 0.1
H0[1, 0] = t_imp
H0[0, 1] = t_imp

# V
V = np.ones((2,2), dtype=np.complex128)
#V = np.identity(2)

# Hbath
eps_bath = np.array([0.2, -0.1], dtype=np.complex128)
#eps_bath = np.array([0.0, 0.0], dtype=np.complex128)
Hbath = np.diag(eps_bath)

beta = 2.0
U = 1.0

asymU = hubbard_asymmU(U)

ed = VertexEvaluatorED(beta, H0, asymU, Hbath, V)
