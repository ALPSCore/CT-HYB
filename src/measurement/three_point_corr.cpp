#include "three_point_corr.ipp"

typedef SlidingWindowManager<REAL_EIGEN_BASIS_MODEL> SW_REAL_MATRIX;
typedef SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL> SW_COMPLEX_MATRIX;

template class ThreePointCorrMeas<double,SW_REAL_MATRIX,PH_CHANNEL>;
template class ThreePointCorrMeas<std::complex<double>,SW_COMPLEX_MATRIX,PH_CHANNEL>;

template class ThreePointCorrMeas<double,SW_REAL_MATRIX,PP_CHANNEL>;
template class ThreePointCorrMeas<std::complex<double>,SW_COMPLEX_MATRIX,PP_CHANNEL>;

template<>
std::string
ThreePointCorrMeasGetNameHelper<PH_CHANNEL>::operator()() const {
  return "eta";
}

template<>
std::string
ThreePointCorrMeasGetNameHelper<PP_CHANNEL>::operator()() const {
  return "gamma";
}