#include <tuple>
#include <alps/params.hpp>

# include "test_sw.hpp"

TEST(SlidingWindow, tau_edges) {
  using MODEL = REAL_EIGEN_BASIS_MODEL;
  double beta = 10.0;
  MODEL* p_model = nullptr;
  SlidingWindowManager<MODEL> sw(p_model, beta);
}