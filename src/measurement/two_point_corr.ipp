#include "two_point_corr.hpp"

template <typename SCALAR, typename SW_TYPE, typename CHANNEL>
void 
TwoPointCorrMeas<SCALAR,SW_TYPE,CHANNEL>::measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements)
{
  using EX_SCALAR = typename SW_TYPE::EXTENDED_SCALAR;

  auto beta = sliding_window.get_beta();

  // Legendre
  auto nl = legendre_trans_.num_legendre();
  std::vector<double> sqrt_2l_1 = legendre_trans_.get_sqrt_2l_1();
  double tau_mod, dummy;
  double tau1 = mc_config.p_worm->get_time(0);
  double tau2 = mc_config.p_worm->get_time(1);
  std::tie(tau_mod, dummy) = fermionic_sign_time_ordering(tau1-tau2, beta);
  std::vector<double> Pl(nl);
  legendre_trans_.compute_legendre(2*tau_mod/beta-1.0, Pl);
  int f0 = mc_config.p_worm->get_flavor(0);
  int f1 = mc_config.p_worm->get_flavor(1);
  int f2 = mc_config.p_worm->get_flavor(2);
  int f3 = mc_config.p_worm->get_flavor(3);
  boost::multi_array<std::complex<double>,5>
    corr_l(boost::extents[nl][nflavors_][nflavors_][nflavors_][nflavors_]);
  std::fill(corr_l.origin(), corr_l.origin()+corr_l.num_elements(), 0.0);
  for (auto l=0; l<nl; ++l) {
    corr_l[l][f0][f1][f2][f3] += (sqrt_2l_1[l] * Pl[l]) * mc_config.sign;
  }

  measure_simple_vector_observable<std::complex<double>>(
    measurements, (get_name()+"_legendre").c_str(), to_std_vector(corr_l));
};