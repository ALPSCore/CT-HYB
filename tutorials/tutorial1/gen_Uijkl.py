import numpy as np
import sys

up = 0
down = 1

def complex_to_str(z):
    return "(%e,%e)"%(z.real,z.imag)

#See Shinaoka (2015): negative sign problem paper
def generate_U_tensor_SK(n_orb, U, JH):
    U_tensor = np.zeros((n_orb,2,n_orb,2,n_orb,2,n_orb,2),dtype=complex)

    num_elem = 0
    for iorb1 in xrange(n_orb):
        for iorb2 in xrange(n_orb):
            for iorb3 in xrange(n_orb):
                for iorb4 in xrange(n_orb):
                    coeff = 0.0
                    if iorb1==iorb2 and iorb2==iorb3 and iorb3==iorb4:
                        coeff = U
                    elif iorb1==iorb4 and iorb2==iorb3 and iorb1!=iorb2:
                        coeff = U-2*JH
                    elif iorb1==iorb3 and iorb2==iorb4 and iorb1!=iorb2:
                        coeff = JH
                    elif iorb1==iorb2 and iorb3==iorb4 and iorb1!=iorb3:
                        coeff = JH

                    for isp in xrange(2):
                        for isp2 in xrange(2):
                            U_tensor[iorb1,isp,    iorb2,isp2,    iorb3,isp2,  iorb4,isp] += coeff
                            if coeff != 0.0:
                                num_elem += 1

    return U_tensor, num_elem

n_site = 3
Uval = 1.0
Jval = 0.25*Uval

V_mat = np.identity(2*n_site,dtype=complex)

U_tensor, num_elem = generate_U_tensor_SK(n_site, Uval, Jval)

f = open("Uijkl.txt", "w")
print >>f, num_elem
line = 0
for iorb1 in xrange(n_site):
    for iorb2 in xrange(n_site):
        for iorb3 in xrange(n_site):
            for iorb4 in xrange(n_site):
                for isp in xrange(2):
                    for isp2 in xrange(2):
                        if U_tensor[iorb1,isp,iorb2,isp2,iorb3,isp2,iorb4,isp] != 0.0:
                            print >>f, line, "   ", 2*iorb1+isp, 2*iorb2+isp2, 2*iorb3+isp2, 2*iorb4+isp, U_tensor[iorb1,isp,iorb2,isp2,iorb3,isp2,iorb4,isp].real, U_tensor[iorb1,isp,iorb2,isp2,iorb3,isp2,iorb4,isp].imag
                            line += 1

f.close()
