#include "impurity.hpp"

//Actual implementation
#include "./impurity.ipp"
#include "./impurity_init.ipp"
#include "./impurity_postprocess.ipp"

template class HybridizationSimulation<ImpurityModelEigenBasis<double> >;
template class HybridizationSimulation<ImpurityModelEigenBasis<std::complex<double> > >;
