#include "vartheta.hpp"


template <typename SW_TYPE>
typename SW_TYPE::EXTENDED_SCALAR
compute_trace_worm_impl(SW_TYPE &sw, const std::vector<psi> &ops) {
  for (const auto& op: ops) {
    sw.insert(op);
  }
  auto trace = sw.compute_trace();
  for (const auto& op: ops) {
    sw.erase(op);
  }
  return trace;
}

template <typename SCALAR, typename SW_TYPE>
void 
VarThetaMeas<SCALAR,SW_TYPE>::measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements)
{
  auto beta = sliding_window.get_beta();

  if (mc_config.p_worm->get_config_space() != G1)
  {
    throw std::runtime_error("Must be measured in G1 space!");
  }
  auto trace_org = mc_config.trace;

  auto sw_wrk(sliding_window);
  sw_wrk.move_edges_to(sw_wrk.get_n_section(), 0);

  // Remove worm operators from the trace
  for (const auto& op : mc_config.p_worm->get_operators()) {
    sw_wrk.erase(op);
  }

  boost::multi_array<std::complex<double>,2> obs(boost::extents[nflavors_][nflavors_]);
  std::fill(obs.origin(), obs.origin()+obs.num_elements(), 0.0);
  auto tau_c     = mc_config.p_worm->get_time(0);
  auto tau_cdagg = mc_config.p_worm->get_time(1);
  double sum_trans_prop = 0.0;
  using EX_SCALAR = typename SW_TYPE::EXTENDED_SCALAR;
  for (auto f0=0; f0<nflavors_; ++f0) {
    for (auto f1=0; f1<nflavors_; ++f1) {
      EX_SCALAR trace_d = compute_trace_worm_impl(sw_wrk,
        std::vector<psi>{
          psi(OperatorTime(tau_c,     0), ANNIHILATION_OP, f0, false),
          psi(OperatorTime(tau_cdagg, 0), CREATION_OP,     f1, false)
        }
      );
      EX_SCALAR trace_q = compute_trace_worm_impl(sw_wrk,
        std::vector<psi>{
          psi(OperatorTime(tau_c,     0), ANNIHILATION_OP, f0, true),
          psi(OperatorTime(tau_cdagg, 0), CREATION_OP,     f1, true)
        }
      );
      sum_trans_prop += static_cast<double>(myabs(trace_d/trace_org));
      obs[f0][f1] = 
        static_cast<SCALAR>(static_cast<EX_SCALAR>(trace_q/trace_org))
        * mc_config.sign;
    }
  }

  std::transform(
    obs.origin(), obs.origin()+obs.num_elements(),
    obs.origin(),
    [&sum_trans_prop](const auto &x) {return x/sum_trans_prop;}
  );

  boost::multi_array<std::complex<double>,3> matsu(boost::extents[vsample_.size()][nflavors_][nflavors_]);
  auto expix = [](double x) {return std::complex<double>(std::cos(x), std::sin(x));};
  auto tau = tau_c - tau_cdagg;
  auto temperature = 1/beta_;
  for (auto n=0; n<vsample_.size(); ++n) {
    auto exp_ = expix(vsample_[n] * M_PI * temperature * tau);
    for (auto f0=0; f0<nflavors_; ++f0) {
      for (auto f1=0; f1<nflavors_; ++f1) {
        matsu[n][f0][f1] = exp_ * obs[f0][f1];
      }
    }
  }

  measure_simple_vector_observable<std::complex<double>>(measurements, "VarTheta", to_std_vector(matsu));
};