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
    return asymmU

def mk_asymm(U):
    """ Antisymmetrize Coulomb tensor"""
    assert U.ndim == 4
    asymmU = 0.5 * (U - U.transpose((1, 0, 2, 3)))
    return 0.5 * (asymmU - asymmU.transpose((0, 1, 3, 2)))

def slater_kanamori_asymm(norb, U, J):
    """Rotationally invariant Slater-Kanamori interaction, inner-most loop is spin"""
    nsp = 2
    tensorU = np.zeros(4*(norb,nsp,))
    for iorb1, iorb2, iorb3, iorb4 in product(range(norb), repeat=4):#eq(2)
        coeff = 0.0
        if iorb1 == iorb4 and iorb2 == iorb3 and iorb1 == iorb2:#aaaa
            coeff = U
        if iorb1 == iorb4 and iorb2 == iorb3 and iorb1 != iorb2:#abab
            coeff = U - 2*J
        if iorb1 == iorb3 and iorb2 == iorb4 and iorb1 != iorb2 :#abba            
            coeff = J
        if iorb1 == iorb2 and iorb3 == iorb4 and iorb2 != iorb4 :#aabb
            coeff = J
        for isp1, isp2 in product(range(nsp), repeat=2):
            tensorU[iorb1,isp1, iorb2,isp2, iorb4, isp1, iorb3,isp2] += coeff
    return mk_asymm(tensorU.reshape(4*(nsp*norb,)))