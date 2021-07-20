#include <tuple>

#include "test_common.hpp"

TEST(Common, is_fermionic) {
  std::vector<int> v1 {-1, 1, 3, 5};
  ASSERT_TRUE(is_fermionic(v1.begin(), v1.end()));

  std::vector<int> v2 {1, 3, 0};
  ASSERT_FALSE(is_fermionic(v2.begin(), v2.end()));
}