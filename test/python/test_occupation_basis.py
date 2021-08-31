from alpscthyb.occupation_basis import *
import pytest

def test_count_n_left():
    assert count_n_left(0b0011, 4, 0) == 1
    assert count_n_left(0b0111, 4, 0) == 2
    assert count_n_left(0b1101, 4, 1) == 2


test_params = [
        (1, False),
        (2, False),
        (3, False),
        (1, True),
        (2, True),
        (3, True),
        ]
@pytest.mark.parametrize("nflavors, sort", test_params)
def test_construct_cdagger_ops(nflavors, sort):
    ntot, cdag_ops = construct_cdagger_ops(nflavors)
    if sort:
        ntot, cdag_ops = sort_by_ntot(ntot, cdag_ops)
    
    c_ops = [op.transpose(copy=True) for op in cdag_ops]
    dim = 2**nflavors

    if sort:
        # Check if ntot is in non-desecnding order
        assert all(ntot[0:-1] <= ntot[1:])

    # Check particle number
    ntot_op = coo_matrix((dim, dim), dtype=np.float64)
    for f in range(nflavors):
        ntot_op += cdag_ops[f] @ c_ops[f]
    assert np.allclose(ntot_op.diagonal(), ntot, atol=1e-8)

    # Check anticommunicator
    for f0 in range(nflavors):
        for f1 in range(nflavors):
            anticom = (c_ops[f0] @ cdag_ops[f1] + cdag_ops[f1] @ c_ops[f0]).toarray()
            if f0 == f1:
               assert np.allclose(anticom, np.identity(dim), atol=1e-8)
            else:
               assert np.allclose(anticom, np.zeros((dim, dim)), atol=1e-8)
