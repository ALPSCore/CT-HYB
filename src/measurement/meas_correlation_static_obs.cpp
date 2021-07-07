#include <Eigen/Core> // Has to be included first to avoid compilation failures(Eigen 3.3.4,Boost 1.65)
                      // see discussion at https://github.com/ALPSCore/CT-HYB/issues/13
#include "../sliding_window/sliding_window.hpp"
#include "meas_correlation_static_obs.hpp"
#include "meas_static_obs.ipp"
#include "meas_correlation.ipp"

/**
 * Real-number version
 */
template
class MeasStaticObs<SlidingWindowManager<REAL_EIGEN_BASIS_MODEL>, CdagC>;
template
class MeasCorrelation<SlidingWindowManager<REAL_EIGEN_BASIS_MODEL>, EqualTimeOperator<1> >;

/**
 * Complex-number version
 */
template
class MeasStaticObs<SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL>, CdagC>;
template
class MeasCorrelation<SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL>, EqualTimeOperator<1> >;


