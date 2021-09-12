#pragma once

#include <gtest.h>

#include "../src/moves/worm.hpp"
#include "../src/moves/moves.hpp"

template<int Rank>
void test_gworm() {
  GWorm<Rank> worm;
  for (auto i=0; i<Rank; ++i) {
    worm.set_small_index(i, 2*i);
    ASSERT_EQ(worm.get_small_index(i), 2*i);
  }
}

template<bool TIME_DERIV>
void test_three_point_corr_worm() {
  ThreePointCorrWorm<PP_CHANNEL, TIME_DERIV> worm;

  std::array<double,3> times = {0.1, 0.2, 0.3};
  for (auto i=0; i<times.size(); ++i) {
    worm.set_time(i, times[i]);
  }

  std::array<int,4> flavors = {0, 1, 2, 3};
  for (auto f=0; f<4; ++f) {
    worm.set_flavor(f, flavors[f]);
  }

  for (auto i=0; i<times.size(); ++i) {
    ASSERT_EQ(worm.get_time(i), times[i]);
  }
  for (auto f=0; f<4; ++f) {
    ASSERT_EQ(worm.get_flavor(f), flavors[f]);
  }

  auto ops = worm.get_operators();
  ASSERT_EQ(ops[0].time_deriv(), TIME_DERIV);
  ASSERT_EQ(ops[1].time_deriv(), TIME_DERIV);
  ASSERT_EQ(ops[2].time_deriv(), false);
  ASSERT_EQ(ops[3].time_deriv(), false);
}