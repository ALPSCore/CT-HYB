#include "impurity.hpp"

//Actual implementation
#include "./impurity.ipp"
#include "./impurity_init.ipp"
#include "./impurity_postprocess.ipp"

template class HybridizationSimulation<AtomicModelEigenBasis<double> >;
template class HybridizationSimulation<AtomicModelEigenBasis<std::complex<double> > >;
