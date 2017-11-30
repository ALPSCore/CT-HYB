#include <Eigen/Core> // Has to be included first to avoid compilation failures(Eigen 3.3.4,Boost 1.65)
                      // see discussion at https://github.com/ALPSCore/CT-HYB/issues/13
#include "sliding_window.hpp"
#include "sliding_window.ipp"
#include "meas_static_obs.ipp"
#include "meas_correlation.ipp"

/**
 * Real-number version
 */
template
class SlidingWindowManager<REAL_EIGEN_BASIS_MODEL>;
template
class MeasStaticObs<SlidingWindowManager<REAL_EIGEN_BASIS_MODEL>, CdagC>;
template
class MeasCorrelation<SlidingWindowManager<REAL_EIGEN_BASIS_MODEL>, EqualTimeOperator<1> >;

/**
 * Complex-number version
 */
template
class SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL>;
template
class MeasStaticObs<SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL>, CdagC>;
template
class MeasCorrelation<SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL>, EqualTimeOperator<1> >;


