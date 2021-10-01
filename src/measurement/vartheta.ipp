#include "vartheta.hpp"


/*
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
*/

template <typename SCALAR, typename SW_TYPE>
void 
VarThetaLegendreMeas<SCALAR,SW_TYPE>::measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements)
{
  auto beta = sliding_window.get_beta();
  auto num_meas = 10;

  auto trace_org = mc_config.trace;

  auto sw_wrk(sliding_window);
  sw_wrk.move_edges_to(sw_wrk.get_n_section(), 0);

  // Remove worm operators from the trace
  for (const auto& op : mc_config.p_worm->get_operators()) {
    sw_wrk.erase(op);
  }

  boost::multi_array<std::complex<double>,2> obs(boost::extents[nflavors_][nflavors_]);

   
  std::fill(obs.origin(), obs.origin()+obs.num_elements(), 0.0);
  auto tau_qdagg = mc_config.p_worm->get_time(1);
  double sum_trans_prop = 0.0;
  using EX_SCALAR = typename SW_TYPE::EXTENDED_SCALAR;

  // Move q operator to random positions
  std::vector<double> taus_q(num_meas);
  taus_q[0] = mc_config.p_worm->get_time(0);
  for (auto imeas=1; imeas < num_meas; ++imeas) {
    taus_q[imeas] = p_rng_->operator()() * beta_;
  }

  // Compute weights
  auto nl = legendre_trans_.num_legendre();
  boost::multi_array<std::complex<double>,3> corr_l(boost::extents[nl][nflavors_][nflavors_]);
  std::fill(corr_l.origin(), corr_l.origin()+corr_l.num_elements(), 0.0);
  std::vector<double> sqrt_2l_1 = legendre_trans_.get_sqrt_2l_1();
  std::vector<double> Pl(nl);
  for (auto imeas=0; imeas < num_meas; ++imeas) {
    auto p_worm_ = mc_config.p_worm->clone();
    p_worm_->set_time(0, taus_q[imeas]);
    EX_SCALAR trace_q = compute_trace_worm_impl(sw_wrk, p_worm_->get_operators());

    int perm_sign = compute_permutation_sign_impl(
       mc_config.M.get_cdagg_ops(),
       mc_config.M.get_c_ops(),
       p_worm_->get_operators()
    );

    sum_trans_prop += static_cast<double>(myabs(trace_q/trace_org));
    auto obs = static_cast<SCALAR>(static_cast<EX_SCALAR>(trace_q/trace_org)) * mc_config.sign;

    double tau, tau_sign;
    std::tie(tau, tau_sign) = fermionic_sign_time_ordering(
      p_worm_->get_time(0)-p_worm_->get_time(1), beta);
    legendre_trans_.compute_legendre(2*tau/beta-1.0, Pl);
    for (auto l=0; l<nl; ++l) {
      corr_l[l][p_worm_->get_flavor(0)][p_worm_->get_flavor(1)] = 
          tau_sign * (sqrt_2l_1[l] * Pl[l]) * obs;
    }
  }

  std::transform(
    corr_l.origin(), corr_l.origin()+corr_l.num_elements(),
    corr_l.origin(),
    [&sum_trans_prop](const auto &x) {return x/sum_trans_prop;}
  );

  measure_simple_vector_observable<std::complex<double>>(
    measurements, "vartheta_legendre", to_std_vector(corr_l));
};