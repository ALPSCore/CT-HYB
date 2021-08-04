#include <tuple>

#include "test_meas.hpp"

TEST(Meas, equal_time_G1_double) {
  test_equal_time_G1<double>();
}


TEST(Meas, fermionic_sign_time_ordering) {
  double beta = 10.0;
  double dtau = 1.0;
  double tau, tau_sign;

  for (auto ishift=-5; ishift < 5; ++ishift) {
    std::tie(tau, tau_sign) = fermionic_sign_time_ordering(dtau + ishift*beta, beta);
    ASSERT_NEAR(dtau, tau, 1e-8);
    ASSERT_NEAR(1.0*std::pow(-1, ishift), tau_sign, 1e-8);
  }
}