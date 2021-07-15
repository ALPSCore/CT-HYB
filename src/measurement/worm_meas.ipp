#include "worm_meas.hpp"

template <typename SCALAR, typename SW_TYPE>
void
EqualTimeG1Meas<SCALAR,SW_TYPE>::postprocess(
    const alps::accumulators::result_set &results,
    double sign, double worm_space_vol_rat,
    std::map<std::string,boost::any> &ar,
    bool verbose)
{
  double temperature(1.0 / beta_);
  double coeff = worm_space_vol_rat / sign;

  std::vector<double> data_Re = results["Equal_time_G1_Re"].template mean<std::vector<double> >();
  std::vector<double> data_Im = results["Equal_time_G1_Im"].template mean<std::vector<double> >();
  check_true(data_Re.size() == nflavors_ * nflavors_);
  boost::multi_array<std::complex<double>, 2> data(boost::extents[nflavors_][nflavors_]);
  std::transform(data_Re.begin(), data_Re.end(), data_Im.begin(), data.origin(), to_complex<double>());
  std::transform(data.origin(), data.origin() + data.num_elements(), data.origin(),
                 std::bind1st(std::multiplies<std::complex<double> >(), coeff));
  ar["EQUAL_TIME_G1"] = data;
}

template <typename SCALAR, typename SW_TYPE>
void 
EqualTimeG1Meas<SCALAR,SW_TYPE>::measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements)
{
  auto beta = sliding_window.get_beta();

  MonteCarloConfiguration<SCALAR> mc_config_wrk(mc_config);
  if (mc_config_wrk.p_worm->get_config_space() != G1)
  {
    throw std::runtime_error("Must be measured in G1 space!");
  }
  mc_config_wrk.p_worm.reset();

  std::vector<double> taus_ins(num_ins_);
  for (auto t = 0; t < num_ins_; ++t)
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
    taus_ins[t] = new_tau;
  }
  std::sort(taus_ins.begin(), taus_ins.end());

  // Ste up a sliding window
  std::vector<double> taus_edges;
  taus_edges.push_back(0.0);
  for (auto& t: taus_ins) {
    taus_edges.push_back(t);
  }
  taus_edges.push_back(sliding_window.get_beta());

  auto sw_wrk(sliding_window);
  sw_wrk.set_mesh(taus_edges, 0, ITIME_LEFT, taus_edges.size()-1); // left_pos = beta, right_pos = 0
  auto trace_org = sw_wrk.compute_trace();
  for (auto op : mc_config.p_worm->get_operators())
  {
    sw_wrk.erase(op);
  }
  sw_wrk.move_edges_to(1, 1);

  std::vector<psi> ops_tmp;
  ops_tmp.emplace_back(OperatorTime(taus_ins[0], 1), CREATION_OP, 0);
  ops_tmp.emplace_back(OperatorTime(taus_ins[0], 0), ANNIHILATION_OP, 1);
  auto perm_sign_change =
      static_cast<double>(mc_config.perm_sign) *
      compute_permutation_sign_impl(
          mc_config.M.get_cdagg_ops(),
          mc_config.M.get_c_ops(),
          ops_tmp
          );

  boost::multi_array<std::complex<double>,2> weight_rat(boost::extents[nflavors_][nflavors_]);
  std::fill(weight_rat.origin(), weight_rat.origin()+weight_rat.num_elements(), 0.0);
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
        SCALAR trace_rat_ = static_cast<SCALAR>(
              static_cast<typename SW_TYPE::EXTENDED_SCALAR>(sw_wrk.compute_trace()/trace_org)
            );
        weight_rat[f0][f1] += perm_sign_change * trace_rat_;
        sw_wrk.erase(c_op);
        sw_wrk.erase(cdagg_op);
      }
    }

    sw_wrk.move_window_to_next_position();
  }

  std::transform(
    weight_rat.origin(), weight_rat.origin()+weight_rat.num_elements(),
    weight_rat.origin(),
    [&](auto x) {return x/(beta*beta*num_ins_);}
  );

  measure_simple_vector_observable<std::complex<double>>(
    measurements, "Equal_time_G1", to_std_vector(weight_rat)
  );
};