#include "worm_meas.hpp"


template <typename SCALAR, typename SW_TYPE>
void 
EqualTimeG1Meas<SCALAR,SW_TYPE>::measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements)
{
  auto beta = sliding_window.get_beta();

  if (mc_config.p_worm->get_config_space() != ConfigSpaceEnum::Equal_time_G1)
  {
    throw std::runtime_error("Must be measured in Equal_time_G1 space!");
  }

  std::vector<double> taus_ins {mc_config.p_worm->get_time(0)};
  while (taus_ins.size() < num_meas_)
  {
    auto new_tau = beta * (*p_rng_)();
    while (true)
    {
      if (count_ops_at(sliding_window.get_operators(), new_tau) == 0 &&
          new_tau != 0 &&
          new_tau != beta &&
          std::find(taus_ins.begin(), taus_ins.end(), new_tau) == taus_ins.end())
      {
        break;
      }
      new_tau = beta * (*p_rng_)();
    }
    taus_ins.push_back(new_tau);
  }
  check_true(taus_ins.size() == num_meas_);
  std::sort(taus_ins.begin(), taus_ins.end());

  // Set up a sliding window
  std::vector<double> taus_edges;
  taus_edges.push_back(0.0);
  for (auto& t: taus_ins) {
    taus_edges.push_back(t);
  }
  taus_edges.push_back(sliding_window.get_beta());

  auto sw_wrk(sliding_window);
  sw_wrk.set_mesh(taus_edges, 0, ITIME_LEFT, taus_edges.size()-1); // left_pos = beta, right_pos = 0
  auto trace_org = sw_wrk.compute_trace();
  auto abs_trace_org = myabs(trace_org);
  for (auto op : mc_config.p_worm->get_operators())
  {
    sw_wrk.erase(op);
  }
  sw_wrk.move_edges_to(1, 1);

  //std::vector<psi> ops_tmp;
  //ops_tmp.emplace_back(OperatorTime(taus_ins[0], 1), CREATION_OP, 0);
  //ops_tmp.emplace_back(OperatorTime(taus_ins[0], 0), ANNIHILATION_OP, 0);
  //double perm_sign =
      //1. * compute_permutation_sign_impl(
          //mc_config.M.get_cdagg_ops(),
          //mc_config.M.get_c_ops(),
          //ops_tmp
          //);
  //double perm_sign_rat = perm_sign/mc_config.compute_perm_sign();

  boost::multi_array<std::complex<double>,3> weights(boost::extents[nflavors_][nflavors_][num_meas_]);
  for (auto t = 0; t < taus_ins.size(); ++t)
  {
    auto tau = taus_ins[t];
    check_true(sw_wrk.get_tau_low() == tau);
    check_true(sw_wrk.get_tau_high() == tau);

    for (auto f0=0; f0<nflavors_; ++f0) {
      for (auto f1=0; f1<nflavors_; ++f1) {
        auto cdagg_op = psi(OperatorTime(tau, 1), CREATION_OP, f0);
        auto c_op = psi(OperatorTime(tau, 0), ANNIHILATION_OP, f1);
        sw_wrk.insert(c_op);
        sw_wrk.insert(cdagg_op);
        SCALAR weight_ = mc_config.sign * static_cast<SCALAR>(
              static_cast<typename SW_TYPE::EXTENDED_SCALAR>(
                sw_wrk.compute_trace()/trace_org
              )
            );
        weights[f0][f1][t] = weight_;
        sw_wrk.erase(c_op);
        sw_wrk.erase(cdagg_op);
      }
    }
    sw_wrk.move_window_to_next_position();
  }

  double sum_abs_weights = 0.0;
  for (auto t = 0; t < taus_ins.size(); ++t) {
    for (auto f0=0; f0<nflavors_; ++f0) {
      for (auto f1=0; f1<nflavors_; ++f1) {
        sum_abs_weights += std::abs(weights[f0][f1][t]);
      }
    }
  }


  boost::multi_array<std::complex<double>,2> obs_(boost::extents[nflavors_][nflavors_]);
  std::fill(obs_.origin(), obs_.origin()+obs_.num_elements(), 0.0);
  for (auto t = 0; t < taus_ins.size(); ++t) {
    for (auto f0=0; f0<nflavors_; ++f0) {
      for (auto f1=0; f1<nflavors_; ++f1) {
        obs_[f0][f1] += weights[f0][f1][t]/sum_abs_weights;
      }
    }
  }

  measure_simple_vector_observable<std::complex<double>>(
    measurements, "Equal_time_G1", to_std_vector(obs_)
  );
};