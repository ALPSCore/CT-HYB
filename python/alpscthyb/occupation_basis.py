import numpy as np
from scipy.sparse import coo_matrix

def count_n_left(i, nflavors, flavor):
    """
    Count the number for particles at f > flavor.
    """
    bit_repr = np.right_shift(i, flavor)
    n = 0
    for f in range(flavor+1,nflavors):
        bit_repr = np.right_shift(bit_repr, 1)
        n += np.bitwise_and(bit_repr, 1)
    return n

def construct_cdagger_ops(nflavors):
    """
    Construct sparse matrix representions of creation operators
    """
    dim = 2**nflavors
    nbits = 32
    mask = 1

    # Count number of particles for each occupation basis vector
    ntot = np.empty(dim, dtype=np.int)
    for ib in range(dim):
        bit_repr = np.uint32(ib)
        ntot[ib] = np.bitwise_and(mask, bit_repr)
        for shift in range(nbits-1):
            bit_repr = np.right_shift(bit_repr, 1)
            ntot[ib] += np.bitwise_and(mask, bit_repr)

    cdag_ops = []
    for flavor in range(nflavors):
        row  = np.zeros((dim,), dtype=np.int)
        col  = np.arange(dim)
        data = np.zeros((dim,), dtype=np.float64)
        for right_basis in range(dim):
            mask = np.left_shift(1, flavor)
            if np.bitwise_and(right_basis, mask) != 0:
                continue
            row[right_basis] = np.bitwise_xor(right_basis, mask)
            data[right_basis] = (-1) ** count_n_left(right_basis, nflavors, flavor)
        cdag_ops.append(coo_matrix((data, (row, col)), shape=(dim, dim)))

    return ntot, cdag_ops

def _swap_rows_cols(op, rev_index_array):
    row = np.asarray([rev_index_array[r] for r in op.row])
    col = np.asarray([rev_index_array[r] for r in op.col])
    return coo_matrix((op.data, (row, col)), shape=op.shape)

def sort_by_ntot(ntot, cdag_ops):
    """
    Sort occupation basis vectors by particle number
    """
    idx = np.argsort(ntot, kind='stable') # Use a stable sort
    rev_idx = np.empty_like(idx)
    for i, idx_ in enumerate(idx):
        rev_idx[idx_] = i
    cdag_ops_sorted = [_swap_rows_cols(op, rev_idx) for op in cdag_ops]
    return ntot[idx].copy(), cdag_ops_sorted