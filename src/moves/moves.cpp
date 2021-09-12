#include <Eigen/Core> // Has to be included first to avoid compilation failures(Eigen 3.3.4,Boost 1.65)
                      // see discussion at https://github.com/ALPSCore/CT-HYB/issues/13
#include <alps/mc/random01.hpp>

#include "moves.hpp"
#include "moves.ipp"

typedef SlidingWindowManager<REAL_EIGEN_BASIS_MODEL> SW_REAL_MATRIX;
typedef SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL> SW_COMPLEX_MATRIX;

#undef PP_REAL
#undef PP_EXTENDED_SCALAR
#undef PP_SW
#define PP_REAL double
#define PP_EXTENDED_SCALAR EXTENDED_REAL
#define PP_SW SW_REAL_MATRIX
#include "moves_explicit.def"

#undef PP_REAL
#undef PP_EXTENDED_SCALAR
#undef PP_SW
#define PP_REAL std::complex<double>
#define PP_EXTENDED_SCALAR EXTENDED_COMPLEX
#define PP_SW SW_COMPLEX_MATRIX
#include "moves_explicit.def"

/**
 * Figure what operators are removed and added to the trace.
 * 
 * Input:
 *   hy_op_rem: bath operartors to be removed.
 *   hy_op_add: bath operartors to be added.
 *   worm_op_rem: worm operators to be removed.
 *   worm_op_rem: worm operators to be added.
 * 
 * Output:
 *   op_rem: operators to be removed from the trace
 *   op_add: operators to be added to the trace
 */
void merge_diff(const std::vector<psi> &hyb_op_rem,
                       const std::vector<psi> &hyb_op_add,
                       const std::vector<psi> &worm_op_rem,
                       const std::vector<psi> &worm_op_add,
                       std::vector<psi> &op_rem,
                       std::vector<psi> &op_add) {
  op_rem.resize(0);
  op_add.resize(0);
  std::copy(hyb_op_rem.begin(), hyb_op_rem.end(), std::back_inserter(op_rem));
  std::copy(worm_op_rem.begin(), worm_op_rem.end(), std::back_inserter(op_rem));
  merge_diff_impl(op_rem, op_add, hyb_op_add);
  merge_diff_impl(op_rem, op_add, worm_op_add);
}