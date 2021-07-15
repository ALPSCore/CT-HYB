#include <tuple>

#include "test_moves.hpp"

TEST(Moves, G1Worm) {
  test_gworm<1>();
}

TEST(Moves, G2Worm) {
  test_gworm<2>();
}