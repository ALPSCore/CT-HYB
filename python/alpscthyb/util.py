import numpy as np

def complex_to_float_array(a):
  a = np.ascontiguousarray(a)
  return a.view(float).reshape(a.shape + (2,))

def float_to_complex_array(a):
  a = np.ascontiguousarray(a)
  if np.iscomplexobj(a):
      return a
  return a.view(complex).reshape(a.shape[:-1])

def box(nf, nb, return_conv='full', ravel=True):
    """Return frequency box"""
    if nf % 2 != 0:
        raise ValueError("number of fermionic frequencies must be even")
    if nb % 2 != 1:
        raise ValueError("number of bosonic frequencies must be odd")

    wf = np.arange(-nf+1, nf, 2)
    wb = np.arange(-nb+1, nb, 2)
    return product(wf, wb, return_conv, ravel)
