#include <tuple>

#include "test_moves.hpp"

TEST(Moves, G1Worm) {
  test_gworm<1>();
}

TEST(Moves, G2Worm) {
  test_gworm<2>();
}

TEST(OperatorTime, time_ordering) {
  auto tau = 0.1;
  auto up = 0;
  auto t1 = OperatorTime(tau, 0);
  auto t2 = OperatorTime(tau, 1);
  ASSERT_TRUE(t2 > t1);
  ASSERT_FALSE(t2 == t1);
}