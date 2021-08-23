#include "h_corr.hpp"

#include <boost/filesystem/operations.hpp>
#include <alps/utilities/fs/get_dirname.hpp>
#include <alps/utilities/fs/get_basename.hpp>
#include <alps/utilities/fs/remove_extensions.hpp>

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
void HCorrMeas<SCALAR,SW_TYPE>::save_results(const std::string &filename, const alps::mpi::communicator &comm) const {
  std::string dirname = 
    alps::fs::remove_extensions(filename) + "_" + get_name() + "_results";
   if (comm.rank()==0) {
      if (boost::filesystem::exists(dirname) && 
        !boost::filesystem::is_directory(dirname)) {
        throw std::runtime_error("Please remove " + dirname + "!");
      } else {
        boost::filesystem::create_directory(dirname);
      }
   }
  comm.barrier();

  alps::hdf5::archive oar(
    dirname+"/rank"+std::to_string(comm.rank())+".out.h5", "w");
  worm_config_record_.save(oar, "");
}
