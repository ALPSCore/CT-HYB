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

TEST(SlidingWindow, move) {
  using MODEL = REAL_EIGEN_BASIS_MODEL;
  double beta = 10.0;
  int nflavors = 1;
  int n_window = 10;
  auto p_model = std::shared_ptr<MODEL>(new MODEL(nflavors));
  auto sw = SlidingWindowManager<MODEL>(p_model, beta, n_window);
  operator_container_t ops;

  ASSERT_EQ(2*n_window, sw.get_position_left_edge());
  ASSERT_EQ(0, sw.get_position_right_edge());

  // Move left edge to pos 2 (right edge stays at 0.)
  sw.move_left_edge_to(ops, 2);
  ASSERT_EQ(2, sw.get_position_left_edge());
  ASSERT_EQ(0, sw.get_position_right_edge());

  // Move left edge to pos 0 (right edge stays at 0.)
  sw.move_left_edge_to(ops, 0);
  ASSERT_EQ(0, sw.get_position_left_edge());
  ASSERT_EQ(0, sw.get_position_right_edge());

  // Move left edge to pos 2*n_window, then right edge to the same position.
  // The window width will be 0.
  sw.move_left_edge_to(ops, 2*n_window);
  sw.move_right_edge_to(ops, 2*n_window);
  ASSERT_EQ(2*n_window, sw.get_position_left_edge());
  ASSERT_EQ(2*n_window, sw.get_position_right_edge());

  // Set the move direction to left, move the window twice
  sw.set_direction(ITIME_LEFT);
  sw.move_window_to_next_position(ops);
  ASSERT_EQ(2*n_window-1, sw.get_position_left_edge());
  ASSERT_EQ(2*n_window-1, sw.get_position_right_edge());
  ASSERT_EQ(ITIME_RIGHT, sw.get_direction_move_local_window());

  sw.move_window_to_next_position(ops);
  ASSERT_EQ(2*n_window-2, sw.get_position_left_edge());
  ASSERT_EQ(2*n_window-2, sw.get_position_right_edge());

  // Move right edge to pos 0, then left edge to pos 2.
  // Set the move direction to left, move the window once
  sw.move_right_edge_to(ops, 0);
  sw.move_left_edge_to(ops, 2);
  sw.set_direction(ITIME_RIGHT);
  sw.move_window_to_next_position(ops);
  ASSERT_EQ(3, sw.get_position_left_edge());
  ASSERT_EQ(1, sw.get_position_right_edge());
  ASSERT_EQ(ITIME_LEFT, sw.get_direction_move_local_window());
}