import numpy as np
import sys

up = 0
down = 1

# H_int = (1/2) * \sum_{ijkl} U_{ijkl} c_i^\dagegr c_j^\dagger c_k c_l
def generate_U_tensor_onsite(n_orb, U):
    U_tensor = np.zeros((n_orb,2,n_orb,2,n_orb,2,n_orb,2),dtype=complex)

    for iorb in xrange(n_orb):
        U_tensor[iorb, up, iorb, down, iorb, down, iorb, up] = U
        U_tensor[iorb, down, iorb, up, iorb, up, iorb, down] = U

    return U_tensor, 2*n_orb

n_site = 2
Uval = 4.0

V_mat = np.identity(2*n_site,dtype=complex)

U_tensor, num_elem = generate_U_tensor_onsite(n_site, Uval)

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
