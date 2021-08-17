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

TEST(Meas, G2matsubara_read_freqs) {
  std::vector<std::tuple<int,int,int>> freqs{
    { 0,  0,  0},//fermion, fermion, boson
    { 0, -1,  1},
    {-1,  0, -1}
  };
  {
    std::ofstream ofs("freqs_PH.txt");
    ofs << freqs.size() << std::endl;
    auto i = 0;
    for (auto f: freqs) {
      ofs << i << " "
        << 2*std::get<0>(f)+1 << " " 
        << 2*std::get<1>(f)+1 << " " 
        << 2*std::get<2>(f)   << std::endl;
      ++i;
    }
  }
  auto i = 0;
  for (auto f: read_matsubara_points("freqs_PH.txt")) {
    ASSERT_EQ(freqs[i], f);
    ++i;
  }

  auto smpl = load_smpl_freqs_SIE("freqs_PH.txt");
  for (auto i=0; i<smpl.n_smpl_freqs; ++i) {
    ASSERT_EQ(smpl.v[i],  2*std::get<0>(freqs[i])+1);
    ASSERT_EQ(smpl.vp[i], 2*std::get<1>(freqs[i])+1);
    ASSERT_EQ(smpl.w[i],  2*std::get<2>(freqs[i]));
  }
}