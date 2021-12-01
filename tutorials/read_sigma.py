import numpy as np
from alpscthyb.post_proc import QMCResult
from itertools import product
import argparse

parser = argparse.ArgumentParser(
    usage='$ python3 read_sigma.py seedname num_of_freqs',
)   
parser.add_argument('seedname',
                    action='store',
                    default=None,
                    type=str,
                    help="seedname"
                    )

parser.add_argument('num_of_freqs',
                    action='store',
                    default=None,
                    type=int,
                    help="seedname"
                    )

args = parser.parse_args()
nw = args.num_of_freqs
print("Reading ", end='')
res = QMCResult(args.seedname, verbose=True)
beta = res.beta

# Fermionic sampling frequencies
wfs = 2*np.arange(-nw, nw) + 1

# From symmetric improved estimators
gir_SIE = res.compute_gir_SIE()
giv = res.compute_giv_SIE(wfs)
sigma_iv = res.compute_sigma_iv(giv, wfs)

# Legendre measurement
giv_legendre = res.compute_giv_from_legendre(wfs)
sigma_iv_legendre = res.compute_sigma_iv(giv_legendre, wfs)

with open(args.seedname + "_sigma_iw.txt", "w") as f:
    print("# Column1: Matsubara frequency", file=f)
    print("# Column2: Re of self-energy", file=f)
    print("# Column3: Im of self-energy", file=f)
    for i, j in product(range(res.nflavors), repeat=2):
        for idx_w, w in enumerate(wfs):
            print(w//2,
                sigma_iv[idx_w,i,j].real,
                sigma_iv[idx_w,i,j].imag,
                sigma_iv_legendre[idx_w,i,j].real,
                sigma_iv_legendre[idx_w,i,j].imag,
                file=f
            )