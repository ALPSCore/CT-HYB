import irbasis
import numpy as np

Lambda = 100000
cutoff = 1e-10

b = irbasis.load('F', Lambda)

dim = np.sum(b.sl()/b.sl(0) > cutoff)

print("dim = ", dim)

with open('sparse_sampling.cpp', 'w') as f:
    print("#include <vector>", file=f)
    sp = b.sampling_points_matsubara(dim-1)
    print("std::vector<int> get_wsample_f() {", file=f)
    print("  std::vector<int> wfs_;", file=f)
    for idx, n in enumerate(sp):
        print(f"  wfs_.push_back({2*n+1});", file=f)
    print("  return wfs_;", file=f)
    print("};", file=f)
