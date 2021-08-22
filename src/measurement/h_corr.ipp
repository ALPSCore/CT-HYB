#include "h_corr.hpp"

#include <boost/filesystem/operations.hpp>

template <typename SCALAR, typename SW_TYPE>
void 
HCorrMeas<SCALAR,SW_TYPE>::measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements)
{
  using EX_SCALAR = typename SW_TYPE::EXTENDED_SCALAR;

  // Set up sliding window
  auto sw_wrk(sliding_window);
  sw_wrk.move_edges_to(sw_wrk.get_n_section(), 0);
  for (const auto& op : mc_config.p_worm->get_operators()) {
    sw_wrk.erase(op);
  }
  auto worm_q_ops(mc_config.p_worm->get_operators());
  for (auto &op: worm_q_ops) {
    op.set_time_deriv(true);
  }
  EX_SCALAR trace_q_ = compute_trace_worm_impl(sw_wrk, worm_q_ops);
  std::complex<double> val =
    static_cast<SCALAR>(static_cast<EX_SCALAR>(trace_q_/mc_config.trace))
    * mc_config.sign;
  worm_config_record_.add(*mc_config.p_worm, val);
}


template <typename SCALAR, typename SW_TYPE>
void save_results(const std::string &filename, const alps::mpi::communicator &comm) const {
  for (auto irank=0; irank<comm.size(); ++irank) {
    if (irank == comm.rank()) {
      alps::hdf5::archive oar(filename, "a");
      std::string path = get_name()+"/dataset"+std::to_string(comm.rank());
      if (comm.rank() == 0) {
        oar[get_name()+"/num_dataset"] << comm.size();
      }
      worm_config_record_.save(oar, path);
    }
    comm.barrier();
  }
}
