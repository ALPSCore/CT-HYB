#include "vartheta.hpp"


template <typename SCALAR, typename SW_TYPE>
void 
VarThetaLegendreMeas<SCALAR,SW_TYPE>::measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements)
{
  check_true(mc_config.p_worm->get_config_space() == ConfigSpaceEnum::vartheta);
  auto beta = sliding_window.get_beta();

  auto nl = legendre_trans_.num_legendre();
  std::vector<double> sqrt_2l_1 = legendre_trans_.get_sqrt_2l_1();

  boost::multi_array<std::complex<double>,3> corr_l(boost::extents[nl][nflavors_][nflavors_]);
  std::fill(corr_l.origin(), corr_l.origin()+corr_l.num_elements(), 0.0);
  double tau, tau_sign;
  std::tie(tau, tau_sign) = fermionic_sign_time_ordering(
    mc_config.p_worm->get_time(0)-mc_config.p_worm->get_time(1), beta);
  std::vector<double> Pl(nl);
  legendre_trans_.compute_legendre(2*tau/beta-1.0, Pl);
  int f0 = mc_config.p_worm->get_flavor(0);
  int f1 = mc_config.p_worm->get_flavor(1);
  for (auto l=0; l<nl; ++l) {
    corr_l[l][f0][f1] = tau_sign * (sqrt_2l_1[l] * Pl[l]) * mc_config.sign;
  }

  measure_simple_vector_observable<std::complex<double>>(
    measurements, "vartheta_legendre", to_std_vector(corr_l));
};