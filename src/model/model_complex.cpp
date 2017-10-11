#include <alps/params.hpp>

#include "../operator.hpp"
#include "model.hpp"
#include "model.ipp"
#include "eigenbasis.ipp"

/**
 * Complex-number version
 */
template
class ImpurityModel<std::complex<double>, ImpurityModelEigenBasis<std::complex<double> > >;
template
class ImpurityModelEigenBasis<std::complex<double> >;
template void ImpurityModel<std::complex<double>, ImpurityModelEigenBasis<std::complex<double> > >::apply_op_bra<1>
    (const EqualTimeOperator<1> &op,
     ImpurityModelEigenBasis<std::complex<double> >::BRAKET_T &bra) const;
template void ImpurityModel<std::complex<double>, ImpurityModelEigenBasis<std::complex<double> > >::apply_op_ket<1>
    (const EqualTimeOperator<1> &op,
     ImpurityModelEigenBasis<std::complex<double> >::BRAKET_T &ket) const;
