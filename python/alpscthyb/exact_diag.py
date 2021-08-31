import numpy as np
from scipy.sparse import coo_matrix
from alpscthyb.interaction import check_asymm
from alpscthyb.occupation_basis import construct_cdagger_ops
from itertools import product
from irbasis_x.freq import check_bosonic, check_fermionic

def construct_ham(hopping, asymmU, cdag_ops):
    """
    Create a sparse matrix representation for
        H = \sum_{ij} t_{ij} c^\dagger_i c_j +
            (1/4) * \sum_{ijkl} U_{ikjl} c^\dagger_i c^\dagger_j c_l c_k.
    """
    nflavors = hopping.shape[0]
    asymmU = check_asymm(asymmU)

    # Construct c ops
    c_ops = [op.transpose(copy=True) for op in cdag_ops]

    dim = 2**nflavors

    ham = coo_matrix((dim, dim), dtype=np.complex128)
    for i, j in product(range(nflavors), repeat=2):
        ham += hopping[i,j] * (cdag_ops[i] @ c_ops[j])

    for i, j, k, l in product(range(nflavors), repeat=4):
        ham += 0.25 * asymmU[i,k,j,l] * \
            (cdag_ops[i] @ cdag_ops[j]) @ (c_ops[l] @ c_ops[k])

    #assert np.abs((ham - ham.transpose().conj()).toarray()).max() < \
        #1e-8 * (np.abs(hopping).max() + np.abs(asymmU).max())
    
    return ham

def _to_eigenbasis(A, eigen_vecs):
    return (eigen_vecs.T.conj() @ A.toarray() @ eigen_vecs)

def compute_fermionic_2pt_corr_func(A, B, beta, wfs, eigen_enes, eigen_vecs):
    """
    Compute fermionic two-point correlation function:
        - int_0^\beta e^{iv tau}<T A(tau) B(0)>
    """
    wfs = check_fermionic(wfs)

    poles = (eigen_enes[:, None] - eigen_enes[None, :]).ravel()

    enes_ = eigen_enes - np.min(eigen_enes)

    # Partition function
    Z = np.sum(np.exp(-beta * enes_))

    # Transform operators to the eigenbasis
    A_ = (eigen_vecs.T.conj() @ A.toarray() @ eigen_vecs)
    B_ = (eigen_vecs.T.conj() @ B.toarray() @ eigen_vecs)

    exp_fact = (
        np.exp(-beta*enes_[:,None]) + np.exp(-beta*enes_[None,:])
    )

    coeffs = np.einsum('mn,nm,mn->mn', exp_fact, A_, B_, optimize=True).ravel()/Z

    # Remove poles with zero coefficients
    idx = np.abs(coeffs) > 1e-10 * np.abs(coeffs).max()
    poles = poles[idx]
    coeffs = coeffs[idx]

    iv = 1J*wfs * np.pi/beta

    return np.sum(coeffs[None,:]/(iv[:,None] - poles[None,:]), axis=1)

def compute_bosonic_2pt_corr_func(A, B, beta, wbs, eigen_enes, eigen_vecs):
    """
    Compute bosonic two-point correlation function:
        int_0^\beta e^{iw tau}<T A(tau) B(0)>
    """
    wbs = check_bosonic(wbs)

    poles = (eigen_enes[:, None] - eigen_enes[None, :])

    enes_ = eigen_enes - np.min(eigen_enes)

    # Partition function
    Z = np.sum(np.exp(-beta * enes_))

    # Transform operators to the eigenbasis
    A_ = (eigen_vecs.T.conj() @ A.toarray() @ eigen_vecs)
    B_ = (eigen_vecs.T.conj() @ B.toarray() @ eigen_vecs)

    exp_fact = (
        np.exp(-beta*enes_[:,None]) - np.exp(-beta*enes_[None,:])
    )

    # Non-singular contribution
    coeffs = np.einsum('mn,nm,mn->mn', exp_fact, A_, B_, optimize=True)
    iw = 1J*wbs * np.pi/beta
    iw[wbs==0] += 1e-10
    tmp = coeffs[None,:,:]/(iw[:,None,None] - poles[None,:,:])

    # Singular contribution
    for idx_w in range(wbs.size):
        if wbs[idx_w] != 0:
            continue
        for m, n in product(range(enes_.size), repeat=2):
            if eigen_enes[m] != eigen_enes[n]:
                continue
            tmp[idx_w, m, n] = beta * np.exp(-beta*enes_[n]) * A_[n,m] * B_[m,n]
    return np.sum(tmp, axis=(1,2))/Z



def compute_3pt_corr_func(F1, F2, B, beta, wsample, eigen_enes, eigen_vecs):
    """
    Compute fermionic three-point correlation function:
        f^{vw} = int_0^\beta dtau_1 dtau_2 dtau_3 e^{iv (tau_1-tau_2) + iw * (tau_2-tau_3)}
            <T F_1(tau_1) F_2(tau_2) B(tau_3)>
    This implements Eqs. (C1) and (C2) in PHYSICAL REVIEW B 100, 075119 (2019).
    There is a typo in these equations:
        The sign in the second term must be "-" due to the excanhge of F_1 and F_2.
    """
    wfs = check_fermionic(wsample[0])
    wbs = check_bosonic(wsample[1])
    return _eval_g(F1, F2, B, beta, (wfs,wbs-wfs), eigen_enes, eigen_vecs)

def _eval_g(F1, F2, B, beta, wsample, eigen_enes, eigen_vecs):
    """
    Compute fermionic three-point correlation function:
        g^{vv'} = int_0^\beta dtau_1 dtau_2 dtau_3 e^{iv (tau_1-tau_3) + iv' * (tau_2-tau_3)}
            <T F_1(tau_1) F_2(tau_2) B(tau_3)>
    This implements Eq. (C2) in PHYSICAL REVIEW B 100, 075119 (2019).
    There is a typo: The sign in the second term must be "-" due to the excanhge of F_1 and F_2.
    """
    v = check_fermionic(wsample[0])
    vp = check_fermionic(wsample[1])
    F1_ = _to_eigenbasis(F1, eigen_vecs)
    F2_ = _to_eigenbasis(F2, eigen_vecs)
    B_ = _to_eigenbasis(B, eigen_vecs)

    term1 = _eval_first_term_g(F1_, F2_, B_, beta, (v,vp), eigen_enes, eigen_vecs)
    term2 = _eval_first_term_g(F2_, F1_, B_, beta, (vp,v), eigen_enes, eigen_vecs)

    return np.asarray(term1 - term2)


def _eval_first_term_g(F1, F2, B, beta, wsample, eigen_enes, eigen_vecs):
    """
    Implements the first term of Eq. (C2)
    The singular terms are taken care of.
    """
    v = check_fermionic(wsample[0])
    vp = check_fermionic(wsample[1])

    enes_ = eigen_enes - np.min(eigen_enes)

    # Partition function
    Z = np.sum(np.exp(-beta * enes_))

    iv  = 1J*v*np.pi/beta
    ivp = 1J*vp*np.pi/beta
    Ediff = enes_[:,None] -enes_[None,:]
    expE = np.exp(-beta * enes_)

    # (iv, m, n): (e^{-beta*E_n} + exp^{-beta*E_m})/(iv + E_m - E_n)
    exp_frac_f = lambda iv: (expE[None,:,None] + expE[None,None,:])\
        /(iv[:,None,None] + Ediff[None,:,:])

    # First term in (C4)
    frac11 = 1/(ivp[:,None,None] + Ediff) # wnl
    frac12 = _frac_b(v+vp, beta, enes_)
    frac13 = exp_frac_f(iv)
    term1 = \
        np.einsum('mn,nl,lm,wnl,wml->w', F1, F2, B, frac11, frac12, optimize=True) + \
        np.einsum('mn,nl,lm,wnl,wmn->w', F1, F2, B, frac11, frac13, optimize=True)

    return term1/Z

def _frac_b(wb, beta, enes):
    iw = wb * np.pi/beta
    iw[wb==0] += 1e-10

    # Non-sigular term
    Ediff = enes[:,None] - enes[None,:]
    expE = np.exp(-beta * enes)
    res = (expE[None,None,:] - (expE[None,:,None]))/(iw[:,None,None] + Ediff[None,:,:])

    # take care of singular term
    for idx_w in range(wb.size):
        if wb[idx_w] != 0:
            continue
        for m, n in product(range(enes.size), repeat=2):
            if enes[m] != enes[n]:
                continue
            res[idx_w, m, n] = beta * np.exp(-beta*enes[m])
    
    return res