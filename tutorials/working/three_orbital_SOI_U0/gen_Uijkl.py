
from itertools import product
import numpy as np
import sys

up = 0
down = 1
nsp = 2

def complex_to_str(z):
    return "(%e,%e)"%(z.real,z.imag)

#See Shinaoka (2015): negative sign problem paper
def generate_U_tensor_SK(n_orb, U, JH):
    U_tensor = np.zeros((n_orb,nsp,n_orb,nsp,n_orb,nsp,n_orb,nsp), dtype=complex)

    num_elem = 0
    for iorb1, iorb2, iorb3, iorb4 in product(list(range(n_orb)), repeat=4):
        coeff = 0.0
        if iorb1==iorb2 and iorb2==iorb3 and iorb3==iorb4:
            coeff = U
        elif iorb1==iorb4 and iorb2==iorb3 and iorb1!=iorb2:
            coeff = U-2*JH
        elif iorb1==iorb3 and iorb2==iorb4 and iorb1!=iorb2:
            coeff = JH
        elif iorb1==iorb2 and iorb3==iorb4 and iorb1!=iorb3:
            coeff = JH

        for isp, isp2 in product(list(range(nsp)), repeat=2):
            U_tensor[iorb1, isp, iorb2, isp2, iorb3, isp2, iorb4, isp] += coeff
            if coeff != 0.0:
                num_elem += 1

    return U_tensor, num_elem

n_site = 3
Uval = 0.0
Jval = 0.0*Uval

U_tensor, num_elem = generate_U_tensor_SK(n_site, Uval, Jval)

f = open("Uijkl.txt", "w")
print(num_elem, file=f)
line = 0
for iorb1, iorb2, iorb3, iorb4 in product(list(range(n_site)), repeat=4):
    for isp, isp2 in product(list(range(nsp)), repeat=2):
        if U_tensor[iorb1,isp,iorb2,isp2,iorb3,isp2,iorb4,isp] != 0.0:
            print(line, "   ", 2*iorb1+isp, 2*iorb2+isp2, 2*iorb3+isp2, 2*iorb4+isp, U_tensor[iorb1,isp,iorb2,isp2,iorb3,isp2,iorb4,isp].real, U_tensor[iorb1,isp,iorb2,isp2,iorb3,isp2,iorb4,isp].imag, file=f)
            line += 1
f.close()
