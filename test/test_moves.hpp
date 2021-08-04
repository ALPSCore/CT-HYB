#pragma once

#include <gtest.h>

#include "../src/moves/worm.hpp"

template<int Rank>
void test_gworm() {
  GWorm<Rank> worm;
  for (auto i=0; i<Rank; ++i) {
    worm.set_small_index(i, 2*i);
    ASSERT_EQ(worm.get_small_index(i), 2*i);
  }
}