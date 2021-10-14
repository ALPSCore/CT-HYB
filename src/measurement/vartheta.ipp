#include "vartheta.hpp"


template <typename SCALAR, typename SW_TYPE>
void 
VarThetaLegendreMeas<SCALAR,SW_TYPE>::measure_no_rw(
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

template <typename SCALAR, typename SW_TYPE>
void 
VarThetaLegendreMeas<SCALAR,SW_TYPE>::measure_rw(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements)
{
  auto beta = sliding_window.get_beta();

  throw std::runtime_error("n_rw > 1 is not supported yet!");

  // Times of all operators
  std::set<double> taus;
  taus.insert(mc_config.p_worm->get_time(0));
  taus.insert(mc_config.p_worm->get_time(1));
  for (auto op: mc_config.M.get_cdagg_ops()) {
    taus.insert(op.time().time());
  }
  for (auto op: mc_config.M.get_c_ops()) {
    taus.insert(op.time().time());
  }
  for (auto t=0; t<sliding_window.get_n_section()+1; ++t) {
    taus.insert(sliding_window.get_tau_edge(t));
  }

  // Find positions to where q operator is moved
  std::vector<double> taus_q(n_meas_);
  taus_q[0] = mc_config.p_worm->get_time(0);
  for (auto imeas=1; imeas < n_meas_; ++imeas) {
    while(true) {
      auto tau_ = p_rng_->operator()() * beta_;
      if (std::find(taus.begin(), taus.end(), tau_) == taus.end()) {
        taus_q[imeas] = tau_;
        taus.insert(tau_);
        break;
      }
    }
  }

  // Temporary sliding_window opbject
  auto sw_wrk(sliding_window);
  sw_wrk.move_edges_to(sw_wrk.get_n_section(), 0);
  sw_wrk.erase(mc_config.p_worm->get_operators()[0]);//remove q operator
  {
    std::vector<double> tau_edges_wrk;
    for (auto t=0; t<sw_wrk.get_n_section()+1; ++t) {
      tau_edges_wrk.push_back(sw_wrk.get_tau_edge(t));
    }
    std::copy(taus_q.begin(), taus_q.end(), std::back_inserter(tau_edges_wrk));
    std::sort(tau_edges_wrk.begin(), tau_edges_wrk.end());
    sw_wrk.set_mesh(tau_edges_wrk,  0, ITIME_LEFT, 0); // The both edges to 0
  }

  auto trace_org = mc_config.trace;

  boost::multi_array<std::complex<double>,2> obs(boost::extents[nflavors_][nflavors_]);
  std::fill(obs.origin(), obs.origin()+obs.num_elements(), 0.0);
  auto tau_qdagg = mc_config.p_worm->get_time(1);
  double sum_trans_prop = 0.0;
  using EX_SCALAR = typename SW_TYPE::EXTENDED_SCALAR;
  auto nl = legendre_trans_.num_legendre();
  boost::multi_array<std::complex<double>,3> corr_l(boost::extents[nl][nflavors_][nflavors_]);
  std::fill(corr_l.origin(), corr_l.origin()+corr_l.num_elements(), 0.0);
  std::vector<double> sqrt_2l_1 = legendre_trans_.get_sqrt_2l_1();
  std::vector<double> Pl(nl);
  for (auto imeas=0; imeas < n_meas_; ++imeas) {
    const auto p_worm_ = mc_config.p_worm;

    // Move the edge to the position where the q operator will be inserted
    while(true) {
      if (sw_wrk.get_tau_edge(sw_wrk.get_position_right_edge()) == taus_q[imeas]) {
        break;
      }
      sw_wrk.move_window_to_next_position();
    }

    auto qop = psi(
        OperatorTime(taus_q[imeas], 0),
        ANNIHILATION_OP,
        p_worm_->get_flavor(0), true
      );
    EX_SCALAR trace_q = compute_trace_worm_impl(sw_wrk, std::vector<psi>{qop});

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
    [&sum_trans_prop](const std::complex<double> &x) {return x/sum_trans_prop;}
  );

  measure_simple_vector_observable<std::complex<double>>(
    measurements, "vartheta_legendre", to_std_vector(corr_l));
};
