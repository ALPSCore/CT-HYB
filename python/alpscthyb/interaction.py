from itertools import product
import numpy as np


def check_asymm(asymmU):
    assert asymmU.ndim == 4
    assert np.allclose(asymmU, -asymmU.transpose((2,1,0,3)))
    assert np.allclose(asymmU, -asymmU.transpose((0,3,2,1)))
    return asymmU

def hubbard_asymmU(U):
    """ Antisymmetric Coulomb tensor for single-orbital Hubbard model"""
    asymmU = np.zeros((2,2,2,2))
    up, dn = 0, 1
    asymmU[up,up,dn,dn] = asymmU[dn,dn,up,up] = U
    asymmU[up,dn,dn,up] = asymmU[dn,up,up,dn] = -U
    return check_asymm(asymmU)