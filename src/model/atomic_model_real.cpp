#include <alps/params.hpp>

#include "atomic_model.hpp"

#include "atomic_model_io.ipp"
#include "atomic_model.ipp"
#include "eigenbasis.ipp"

/**
 * Real-number version
 */
template
class AtomicModel<double, AtomicModelEigenBasis<double> >;
template
class AtomicModelEigenBasis<double>;

template void AtomicModel<double, AtomicModelEigenBasis<double> >::apply_op_bra<1>(const EqualTimeOperator<1> &op,
                                                                                       AtomicModelEigenBasis<double>::BRAKET_T &bra)
    const;

template void AtomicModel<double, AtomicModelEigenBasis<double> >::apply_op_ket<1>(const EqualTimeOperator<1> &op,
                                                                                       AtomicModelEigenBasis<double>::BRAKET_T &ket)
    const;