#include <alps/params.hpp>

#include "atomic_model.hpp"
#include "atomic_model.ipp"
#include "eigenbasis.ipp"

/**
 * Complex-number version
 */
template
class AtomicModel<std::complex<double>, AtomicModelEigenBasis<std::complex<double> > >;
template
class AtomicModelEigenBasis<std::complex<double> >;
template void AtomicModel<std::complex<double>, AtomicModelEigenBasis<std::complex<double> > >::apply_op_bra<1>
    (const EqualTimeOperator<1> &op,
     AtomicModelEigenBasis<std::complex<double> >::BRAKET_T &bra) const;
template void AtomicModel<std::complex<double>, AtomicModelEigenBasis<std::complex<double> > >::apply_op_ket<1>
    (const EqualTimeOperator<1> &op,
     AtomicModelEigenBasis<std::complex<double> >::BRAKET_T &ket) const;