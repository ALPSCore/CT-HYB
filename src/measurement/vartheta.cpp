#include "vartheta.ipp"

typedef SlidingWindowManager<REAL_EIGEN_BASIS_MODEL> SW_REAL_MATRIX;
typedef SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL> SW_COMPLEX_MATRIX;

template class VarThetaLegendreMeas<double,SW_REAL_MATRIX>;
template class VarThetaLegendreMeas<std::complex<double>,SW_COMPLEX_MATRIX>;