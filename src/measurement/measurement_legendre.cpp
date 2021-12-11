#include "measurement_legendre.hpp"
#include "measurement_legendre.ipp"

typedef SlidingWindowManager<REAL_EIGEN_BASIS_MODEL> SW_REAL_MATRIX;
typedef SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL> SW_COMPLEX_MATRIX;

//template class Reconnections<double>;
template struct MeasureGHelper<double, 1>;
template struct MeasureGHelper<double, 2>;
template class GLegendreMeasurement<double, SW_REAL_MATRIX, 1>;
template class GLegendreMeasurement<double, SW_REAL_MATRIX, 2>;

//template class Reconnections<std::complex<double>>;
template struct MeasureGHelper<std::complex<double>, 1>;
template struct MeasureGHelper<std::complex<double>, 2>;
template class GLegendreMeasurement<std::complex<double>, SW_COMPLEX_MATRIX, 1>;
template class GLegendreMeasurement<std::complex<double>, SW_COMPLEX_MATRIX, 2>;

