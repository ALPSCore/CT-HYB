#include "two_point_corr.hpp"

template <typename SCALAR, typename SW_TYPE, typename CHANNEL>
void 
TwoPointCorrMeas<SCALAR,SW_TYPE,CHANNEL>::measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements)
{
  using EX_SCALAR = typename SW_TYPE::EXTENDED_SCALAR;
  int N_meas = 100;

  auto beta = sliding_window.get_beta();

  //if (mc_config.p_worm->get_config_space() != G1)
  //{
    //throw std::runtime_error("Must be measured in G1 space!");
  //}
  auto trace_org = mc_config.trace;

  auto sw_wrk(sliding_window);
  sw_wrk.move_edges_to(sw_wrk.get_n_section(), 0);

  // Remove worm operators from the trace
  for (const auto& op : mc_config.p_worm->get_operators()) {
    sw_wrk.erase(op);
  }

  // Legendre
  auto nl = legendre_trans_.num_legendre();
  std::vector<double> sqrt_2l_1 = legendre_trans_.get_sqrt_2l_1();
  boost::multi_array<std::complex<double>,5>
    corr_l(boost::extents[nl][nflavors_][nflavors_][nflavors_][nflavors_]);
  std::fill(corr_l.origin(), corr_l.origin()+corr_l.num_elements(), 0.0);

  boost::multi_array<std::complex<double>,4>
    obs(boost::extents[nflavors_][nflavors_][nflavors_][nflavors_]);
  double sum_trans_prop = 0.0;
  for (auto i_meas=0; i_meas < N_meas; ++i_meas) {
    std::fill(obs.origin(), obs.origin()+obs.num_elements(), 0.0);
    double tau1, tau2;
    if (i_meas == 0) {
      tau1 = mc_config.p_worm->get_time(0);
      tau2 = mc_config.p_worm->get_time(1);
    } else {
      tau1 = beta_ * p_rng_->operator()();
      tau2 = beta_ * p_rng_->operator()();
    }
    for (auto f0=0; f0<nflavors_; ++f0) {
      for (auto f1=0; f1<nflavors_; ++f1) {
        for (auto f2=0; f2<nflavors_; ++f2) {
          for (auto f3=0; f3<nflavors_; ++f3) {
            std::shared_ptr<Worm> p_worm = mc_config.p_worm->clone();
            p_worm->set_flavor(0, f0);
            p_worm->set_flavor(1, f1);
            p_worm->set_flavor(2, f2);
            p_worm->set_flavor(3, f3);
            p_worm->set_time(0, tau1);
            p_worm->set_time(1, tau2);
            EX_SCALAR trace_ = compute_trace_worm_impl(sw_wrk, p_worm->get_operators());
            sum_trans_prop += static_cast<double>(myabs(trace_/trace_org));
            obs[f0][f1][f2][f3] = 
              static_cast<SCALAR>(static_cast<EX_SCALAR>(trace_/trace_org))
              * mc_config.sign;
          }
        }
      }
    }

    // Legendre
    double tau_mod, dummy;
    std::tie(tau_mod, dummy) = fermionic_sign_time_ordering(tau1-tau2, beta);
    std::vector<double> Pl(nl);
    legendre_trans_.compute_legendre(2*tau_mod/beta-1.0, Pl);
    for (auto l=0; l<nl; ++l) {
      for (auto f0=0; f0<nflavors_; ++f0) {
        for (auto f1=0; f1<nflavors_; ++f1) {
          for (auto f2=0; f2<nflavors_; ++f2) {
            for (auto f3=0; f3<nflavors_; ++f3) {
              corr_l[l][f0][f1][f2][f3] += (sqrt_2l_1[l] * Pl[l]) * obs[f0][f1][f2][f3];
            }
          }
        }
      }
    }

  }


  std::transform(
    corr_l.origin(), corr_l.origin()+corr_l.num_elements(),
    corr_l.origin(),
    [&sum_trans_prop](const auto &x) {return x/sum_trans_prop;}
  );

  measure_simple_vector_observable<std::complex<double>>(
    measurements, (get_name()+"_legendre").c_str(), to_std_vector(corr_l));

  // Matsubara
  /*
  boost::multi_array<std::complex<double>,3> matsu(boost::extents[vsample_.size()][nflavors_][nflavors_]);
  auto expix = [](double x) {return std::complex<double>(std::cos(x), std::sin(x));};
  auto tau = tau1 - tau2;
  auto temperature = 1/beta_;
  for (auto n=0; n<vsample_.size(); ++n) {
    auto exp_ = expix(vsample_[n] * M_PI * temperature * tau);
    for (auto f0=0; f0<nflavors_; ++f0) {
      for (auto f1=0; f1<nflavors_; ++f1) {
        matsu[n][f0][f1] = exp_ * obs[f0][f1];
      }
    }
  }
  measure_simple_vector_observable<std::complex<double>>(measurements,
    (get_name()+"_wsample").c_str(), to_std_vector(matsu));
  */
};