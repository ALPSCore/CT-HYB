#include "h_corr.hpp"

template <typename SCALAR, typename SW_TYPE>
void 
HCorrMeas<SCALAR,SW_TYPE>::measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements)
{
  using EX_SCALAR = typename SW_TYPE::EXTENDED_SCALAR;
  //std::vector<std::chrono::high_resolution_clock::time_point> times;
  //times.push_back(std::chrono::high_resolution_clock::now());//0

  auto beta = sliding_window.get_beta();

  auto trace_org = mc_config.trace;

  auto sw_wrk(sliding_window);
  //times.push_back(std::chrono::high_resolution_clock::now());//1
  sw_wrk.move_edges_to(sw_wrk.get_n_section(), 0);
  //times.push_back(std::chrono::high_resolution_clock::now());//2

  // Remove worm operators from the trace
  for (const auto& op : mc_config.p_worm->get_operators()) {
    sw_wrk.erase(op);
  }
  //times.push_back(std::chrono::high_resolution_clock::now());//3

  boost::multi_array<std::complex<double>,5>
    matsu_data(boost::extents[v1_.size()][nflavors_][nflavors_][nflavors_][nflavors_]);
  std::fill(matsu_data.origin(), matsu_data.origin()+matsu_data.num_elements(), 0.0);
  //times.push_back(std::chrono::high_resolution_clock::now());//4

  boost::multi_array<std::complex<double>,4>
    obs(boost::extents[nflavors_][nflavors_][nflavors_][nflavors_]);
  double sum_trans_prop = 0.0;
  for (auto i_meas=0; i_meas < nsmpl_; ++i_meas) {
    //std::cout << "i_meas" << i_meas << std::endl;
    std::fill(obs.origin(), obs.origin()+obs.num_elements(), 0.0);
    std::array<double,3> taus;
    auto ntimes = mc_config.p_worm->num_independent_times();
    // COMPUTE PERMUTATION SIGN
    if (i_meas == 0) {
      for (auto i=0; i<ntimes; ++i) {
        taus[i] = mc_config.p_worm->get_time(i);
      }
    } else {
      for (auto i=0; i<ntimes; ++i) {
        taus[i] = beta_ * p_rng_->operator()();
      }
    }
    std::shared_ptr<Worm> p_worm = mc_config.p_worm->clone();
    for (auto i=0; i<ntimes; ++i) {
      p_worm->set_time(i, taus[i]);
    }
    int perm_sign = compute_permutation_sign_impl(
      mc_config.M.get_cdagg_ops(),
      mc_config.M.get_c_ops(),
      p_worm->get_operators()
    );
    for (auto f0=0; f0<nflavors_; ++f0) {
      for (auto f1=0; f1<nflavors_; ++f1) {
        for (auto f2=0; f2<nflavors_; ++f2) {
          for (auto f3=0; f3<nflavors_; ++f3) {
            p_worm->set_flavor(0, f0);
            p_worm->set_flavor(1, f1);
            p_worm->set_flavor(2, f2);
            p_worm->set_flavor(3, f3);
            const std::vector<psi>& worm_ops = p_worm->get_operators();
            EX_SCALAR trace_ = compute_trace_worm_impl(sw_wrk, worm_ops);

            auto worm_q_ops(worm_ops);
            for (auto &op: worm_q_ops) {
              op.set_time_deriv(true);
            }
            EX_SCALAR trace_q_ = compute_trace_worm_impl(sw_wrk, worm_q_ops);

            sum_trans_prop += static_cast<double>(myabs(trace_/trace_org));
            obs[f0][f1][f2][f3] = 
              static_cast<SCALAR>(static_cast<EX_SCALAR>(trace_q_/trace_org))
              * mc_config.sign;
          }
        }
      }
    }

    auto expix = [](double x) {return std::complex<double>(std::cos(x), std::sin(x));};
    auto temperature = 1/beta_;
    auto sign_change = static_cast<double>(perm_sign/mc_config.perm_sign);
    for (auto n=0; n<v1_.size(); ++n) {
      auto tau_f = taus[0] - taus[1];
      auto tau_b = taus[1] - taus[2];
      auto exp_ = expix(
        M_PI * temperature * 
        (+v1_[n]*taus[0]
         -v2_[n]*taus[1]
         +v3_[n]*taus[2]
         -v4_[n]*taus[3])
      );
      for (auto f0=0; f0<nflavors_; ++f0) {
        for (auto f1=0; f1<nflavors_; ++f1) {
          for (auto f2=0; f2<nflavors_; ++f2) {
            for (auto f3=0; f3<nflavors_; ++f3) {
              matsu_data[n][f0][f1][f2][f3] += sign_change * exp_ * obs[f0][f1][f2][f3];
            }
          }
        }
      }
    }

  }//i_meas
  //times.push_back(std::chrono::high_resolution_clock::now());//5

  std::transform(
    matsu_data.origin(), matsu_data.origin()+matsu_data.num_elements(),
    matsu_data.origin(),
    [&sum_trans_prop](const auto &x) {return x/sum_trans_prop;}
  );
  //times.push_back(std::chrono::high_resolution_clock::now());

  measure_simple_vector_observable<std::complex<double>>(
    measurements, "h_corr", to_std_vector(matsu_data));
  //times.push_back(std::chrono::high_resolution_clock::now());
  //for(auto i=0; i<times.size()-1; ++i) {
    //logger_out << "timing " << i << " " << std::chrono::duration_cast<std::chrono::milliseconds>(times[i+1] - times[i]).count() << std::endl;
  //}
};