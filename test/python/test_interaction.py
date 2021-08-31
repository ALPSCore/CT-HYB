import numpy as np

from alpscthyb.interaction import *

def test_hubbard_U():
    U = 2.0
    nso = 2
    asymmU = check_asymm(hubbard_asymmU(U))

    # Fully occupied case
    mf = np.einsum('ikjl,kl->ij', asymmU, np.identity(nso))
    np.testing.assert_allclose(mf, U*np.identity(nso))
