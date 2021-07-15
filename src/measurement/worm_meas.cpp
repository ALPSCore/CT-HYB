#include "worm_meas.ipp"

typedef SlidingWindowManager<REAL_EIGEN_BASIS_MODEL> SW_REAL_MATRIX;
typedef SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL> SW_COMPLEX_MATRIX;

template class EqualTimeG1Meas<double,SW_REAL_MATRIX>;
template class EqualTimeG1Meas<std::complex<double>,SW_COMPLEX_MATRIX>;