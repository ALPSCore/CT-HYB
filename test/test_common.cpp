#include <tuple>

#include "test_common.hpp"

TEST(Common, is_fermionic) {
  std::vector<int> v1 {-1, 1, 3, 5};
  ASSERT_TRUE(is_fermionic(v1.begin(), v1.end()));

  std::vector<int> v2 {1, 3, 0};
  ASSERT_FALSE(is_fermionic(v2.begin(), v2.end()));
}

namespace ConfigSpaceEnum{
  enum Type {
    Z_FUNCTION,
    G1,
    G2,
    Equal_time_G1,
    Equal_time_G2,
    lambda,
    varphi,
    Three_point_PH,
    Three_point_PP,
    vartheta,
    eta,
    gamma,
    h,
    Unknown
  };
}

TEST(Common, unique) {
  std::vector<int> x {ConfigSpaceEnum::G1, ConfigSpaceEnum::vartheta, ConfigSpaceEnum::G2, ConfigSpaceEnum::vartheta};
  auto x_unique = unique(x);
  ASSERT_TRUE(x_unique.size() == 3);
}