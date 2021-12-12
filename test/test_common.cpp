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

TEST(Common, legendre_basis) {
  double beta = 10.0;
  int size = 2;
  LegendreBasis basis(FERMION, beta, size);
  std::vector<double> xs {-1.0, 0.1, 1.0};
  std::vector<double> val(size);
  for (auto x: xs) {
    basis.value(0.5*beta*(x+1), val);
    ASSERT_NEAR(std::sqrt(2*0+1)/beta, val[0], 1e-10); //l=0
    ASSERT_NEAR((std::sqrt(2*1+1)/beta) * x, val[1], 1e-10); //l=1
  }
}

TEST(Common, irbasis) {
  double beta = 10.0;
  double wmax = 100.0;
  IrBasis basis(FERMION, beta, wmax*beta, 10000, "./irbasis.h5");

  std::vector<double> val(basis.dim());
  auto tau = beta;
  basis.value(tau, val);
  ASSERT_NEAR(val[0], 2.716842916855695, 1e-10);
  ASSERT_NEAR(val[1], 3.634405259826317, 1e-10);
  //import irbasis3
  //b = irbasis3.FiniteTempBasis(irbasis3.KernelFFlat(1000), "F", 10)
  //b.u(beta)
}