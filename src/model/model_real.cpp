#include <alps/params.hpp>

#include "../operator.hpp"
#include "model.hpp"
#include "model.ipp"
#include "eigenbasis.ipp"

/**
 * Real-number version
 */
template
class ImpurityModel<double, ImpurityModelEigenBasis<double> >;
template
class ImpurityModelEigenBasis<double>;
template void ImpurityModel<double, ImpurityModelEigenBasis<double> >::apply_op_bra<1>(const EqualTimeOperator<1> &op,
                                                                                       ImpurityModelEigenBasis<double>::BRAKET_T &bra)
    const;
template void ImpurityModel<double, ImpurityModelEigenBasis<double> >::apply_op_ket<1>(const EqualTimeOperator<1> &op,
                                                                                       ImpurityModelEigenBasis<double>::BRAKET_T &ket)
    const;
