#include <Eigen/Core> // Has to be included first to avoid compilation failures(Eigen 3.3.4,Boost 1.65)
                      // see discussion at https://github.com/ALPSCore/CT-HYB/issues/13
#include "sliding_window.hpp"
#include "sliding_window.ipp"

/**
 * Real-number version
 */
template
class SlidingWindowManager<REAL_EIGEN_BASIS_MODEL>;

/**
 * Complex-number version
 */
template
class SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL>;