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

TEST(Moves, ThreePointCorrWorm) {
    test_three_point_corr_worm<true>();
}


// Convert two bath operators to worm operators
// Something like adding a G1 worm.
TEST(Moves, merge_diff1) {
  auto flavor = 0;
  auto t1 = 0.1;
  auto t2 = 0.2;

  std::vector<psi> hyb_op_rem = {
    psi(OperatorTime(t1), CREATION_OP, flavor),
    psi(OperatorTime(t2), ANNIHILATION_OP, flavor)
  };
  std::vector<psi> worm_op_rem = {};

  std::vector<psi> hyb_op_add = {};
  std::vector<psi> worm_op_add = {
    psi(OperatorTime(t1), CREATION_OP, flavor),
    psi(OperatorTime(t2), ANNIHILATION_OP, flavor)
  };

  std::vector<psi> op_rem, op_add;
  merge_diff(hyb_op_rem, hyb_op_add, worm_op_rem, worm_op_add, op_rem, op_add);

  ASSERT_EQ(0, op_rem.size());
  ASSERT_EQ(0, op_add.size());
}

// Convert two bath operators to worm operators with time_derive = true
// Something like adding a vartheta worm.
TEST(Moves, merge_diff2) {
  auto flavor = 0;
  auto t1 = 0.1;
  auto t2 = 0.2;

  std::vector<psi> hyb_op_rem = {
    psi(OperatorTime(t1), CREATION_OP, flavor),
    psi(OperatorTime(t2), ANNIHILATION_OP, flavor)
  };
  std::vector<psi> worm_op_rem = {};

  std::vector<psi> hyb_op_add = {};
  std::vector<psi> worm_op_add = {
    psi(OperatorTime(t1), CREATION_OP, flavor, true),
    psi(OperatorTime(t2), ANNIHILATION_OP, flavor, true)
  };

  std::vector<psi> op_rem, op_add;
  merge_diff(hyb_op_rem, hyb_op_add, worm_op_rem, worm_op_add, op_rem, op_add);

  ASSERT_EQ(hyb_op_rem, op_rem);
  ASSERT_EQ(worm_op_add, op_add);
}

// Convert two worm operators with time_derive = true to bath operators
// Something like removing a vartheta worm.
TEST(Moves, merge_diff3) {
  auto flavor = 0;
  auto t1 = 0.1;
  auto t2 = 0.2;

  std::vector<psi> hyb_op_rem = {};
  std::vector<psi> worm_op_rem = {
    psi(OperatorTime(t1), CREATION_OP, flavor, true),
    psi(OperatorTime(t2), ANNIHILATION_OP, flavor, true)
  };

  std::vector<psi> hyb_op_add = {
    psi(OperatorTime(t1), CREATION_OP, flavor),
    psi(OperatorTime(t2), ANNIHILATION_OP, flavor)
  };
  std::vector<psi> worm_op_add = {};

  std::vector<psi> op_rem, op_add;
  merge_diff(hyb_op_rem, hyb_op_add, worm_op_rem, worm_op_add, op_rem, op_add);

  ASSERT_EQ(worm_op_rem, op_rem);
  ASSERT_EQ(hyb_op_add, op_add);
}