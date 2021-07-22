import numpy as np

def complex_to_float_array(a):
  a = np.ascontiguousarray(a)
  return a.view(float).reshape(a.shape + (2,))

def float_to_complex_array(a):
  a = np.ascontiguousarray(a)
  if np.iscomplexobj(a):
      return a
  return a.view(complex).reshape(a.shape[:-1])
