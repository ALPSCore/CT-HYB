import numpy as np
import matplotlib.pyplot as plt
from alpscthyb.post_proc import QMCResult, VertexEvaluatorAtomED
from alpscthyb.util import *

res = QMCResult('input', verbose=True)
beta = res.beta

ed = VertexEvaluatorAtomED(res.nflavors, res.beta, res.hopping, res.get_asymU())

nf = 5
wfs = 2*np.arange(-nf,nf)+1

G2 = res.compute_G2_from_IR(wfs)
