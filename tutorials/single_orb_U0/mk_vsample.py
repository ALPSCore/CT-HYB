import irbasis
import numpy as np

Lambda = 10000
cutoff = 1e-6

b = irbasis.load('F', Lambda)

dim = np.sum(b.sl()/b.sl(0) > cutoff)

print("dim = ", dim)

with open('vsample.txt', 'w') as f:
    print(dim, file=f)
    for idx, n in enumerate(b.sampling_points_matsubara(dim-1)):
        print(idx, 2*n+1, file=f)
