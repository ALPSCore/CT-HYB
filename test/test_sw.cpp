#include <memory>
#include <tuple>
#include <iostream>
#include <alps/params.hpp>

# include "test_sw.hpp"
# include "../src/model/atomic_model.hpp"

TEST(SlidingWindow, tau_edges) {
  using MODEL = REAL_EIGEN_BASIS_MODEL;
  double beta = 10.0;
  int nflavors = 1;
  int n_window = 10;
  auto p_model = std::shared_ptr<MODEL>(new MODEL(nflavors));
  auto sw = SlidingWindowManager<MODEL>(p_model, beta, n_window);

  ASSERT_FLOAT_EQ(sw.get_tau_edge(0), 0.0);
  ASSERT_FLOAT_EQ(sw.get_tau_edge(2*n_window), beta);
  for (auto i=1; i<2*n_window; ++i) {
    ASSERT_NEAR((beta * i)/(2*n_window), sw.get_tau_edge(i), 1e-10);
  }

  // Double n_window
  auto n_window2 = 2 * n_window;
  sw.set_window_size(n_window2);
  ASSERT_FLOAT_EQ(sw.get_tau_edge(0), 0.0);
  ASSERT_FLOAT_EQ(sw.get_tau_edge(2*n_window2), beta);
  for (auto i=1; i<2*n_window; ++i) {
    ASSERT_NEAR((beta * i)/(2*n_window2), sw.get_tau_edge(i), 1e-10);
  }

}