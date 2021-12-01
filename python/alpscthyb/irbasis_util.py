import numpy as np

# Written by Markus Wallerberger

def collect(*arrs):
    """Collect arrays into a single record array, allowing sorting"""
    # pylint: disable=invalid-unary-operand-type
    arrs = tuple(map(np.asarray, arrs))
    digits = 64 // len(arrs)
    maxval = np.int64(1 << (digits - 1))
    shape = np.broadcast(*arrs).shape
    if all(np.issubdtype(a.dtype, np.integer)
           and a.min() >= -maxval and a.max() < maxval for a in arrs):
        # Compress to a single integer
        result = np.empty(shape, np.uint64)
        result[...] = arrs[0] + maxval
        for a in arrs[1:]:
            result <<= digits
            result[...] |= (a + maxval).astype(np.uint64)
        return result
    else:
        # Compress to a record array
        dtype = np.dtype([('', a.dtype) for a in arrs])
        result = np.empty(shape, dtype)
        for field, a in zip(dtype.names, arrs):
            result[field] = a
        return result


def split(coll, n):
    """Split record array into parts"""
    fields = coll.dtype.names
    if not fields:
        # Decompression from a single integer
        digits = 64 // n
        maxval = np.int64(1 << (digits - 1))
        mask = np.uint64((1 << digits) - 1)
        results = []
        coll = coll.copy()
        for _ in range(n):
            r = np.bitwise_and(coll, mask).astype(np.int64) - maxval
            results.append(r)
            coll >>= digits
        return tuple(results[::-1])
    else:
        # Decompression from record array
        if n != len(fields):
            raise ValueError("n inconsistent with # of fields")
        return tuple(coll[field] for field in fields)


def check_fermionic(w):
    """Check `w` is a reduced fermionic frequency, i.e., an odd number"""
    w = np.asarray(w)
    if (w % 2 != 1).any():
        raise ValueError("invalid fermionic reduced frequency")
    return w

def check_bosonic(w):
    """Check `w` is a reduced bosonic frequency, i.e., an even number"""
    w = np.asarray(w)
    if (w % 2).any():
        raise ValueError("invalid bosonic reduced frequency")
    return w


def check_full_convention(w1, w2, w3, w4):
    """Check that a set of frequencies is indeed a set of full frequencies"""
    w1 = check_fermionic(w1)
    w2 = check_fermionic(w2)
    w3 = check_fermionic(w3)
    w4 = check_fermionic(w4)
    return w1, w2, w3, w4