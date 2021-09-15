#include "three_point_corr.hpp"

#include <boost/filesystem/operations.hpp>
#include <alps/utilities/fs/get_dirname.hpp>
#include <alps/utilities/fs/get_basename.hpp>
#include <alps/utilities/fs/remove_extensions.hpp>

template <typename SCALAR, typename SW_TYPE, typename CHANNEL>
void 
ThreePointCorrMeas<SCALAR,SW_TYPE,CHANNEL>::measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements)
{
  using EX_SCALAR = typename SW_TYPE::EXTENDED_SCALAR;

  auto sw_wrk(sliding_window);
  sw_wrk.move_edges_to(sw_wrk.get_n_section(), 0);
  // Remove worm operators from the trace
  for (const auto& op : mc_config.p_worm->get_operators()) {
    sw_wrk.erase(op);
  }
  auto worm_q_ops(mc_config.p_worm->get_operators());
  worm_q_ops[0].set_time_deriv(true);
  worm_q_ops[1].set_time_deriv(true);
  EX_SCALAR trace_q_ = compute_trace_worm_impl(sw_wrk, worm_q_ops);
  std::complex<double> val =
    static_cast<SCALAR>(static_cast<EX_SCALAR>(trace_q_/mc_config.trace))
      * mc_config.sign;
  worm_config_record_.add(*mc_config.p_worm, val);
};



template <typename SCALAR, typename SW_TYPE, typename CHANNEL>
void 
ThreePointCorrMeas<SCALAR,SW_TYPE,CHANNEL>::eval_on_smpl_freqs(
    const std::vector<int> &wfs,
    const std::vector<int> &wbs,
    const std::string &datafile,
    const std::string &outputfile) {
  WormConfigRecord<double,double> worm_config_record(3, 4);

  double beta;
  int nflavors;
  {
    alps::hdf5::archive iar(datafile, "r");
    worm_config_record.load(iar, get_name());
    iar["/parameters/model.beta"] >> beta;
    iar["/parameters/model.flavors"] >> nflavors;
  }

  auto expix = [](double x) {return std::complex<double>(std::cos(x), std::sin(x));};
  auto temperature = 1/beta;

  alps::numerics::tensor<std::complex<double>,5> 
    matsu_data(wfs.size(), nflavors,nflavors,nflavors,nflavors);
  matsu_data.set_number(0.0);

  for (auto smpl=0; smpl<worm_config_record.nsmpl(); ++smpl) {
    auto tau_f = worm_config_record.taus[0][smpl] - worm_config_record.taus[1][smpl];
    auto tau_b = worm_config_record.taus[1][smpl] - worm_config_record.taus[2][smpl];
    auto val = std::complex<double>(
      worm_config_record.vals_real[smpl],
      worm_config_record.vals_imag[smpl]
    );
    for (auto ifreq=0; ifreq<wfs.size(); ++ifreq) {
      auto exp_ = expix(
        M_PI * temperature * (wfs[ifreq]*tau_f + wbs[ifreq]*tau_b)
      );
      matsu_data(ifreq,
        worm_config_record.flavors[0][smpl],
        worm_config_record.flavors[1][smpl],
        worm_config_record.flavors[2][smpl],
        worm_config_record.flavors[3][smpl]) += exp_ * val;
    }
  }

  matsu_data /= (beta * worm_config_record.nsmpl());
  {
    alps::hdf5::archive oar(outputfile, "a");
    oar[get_name()+"_matsubara"] = matsu_data;
  }
}

template <typename SCALAR, typename SW_TYPE, typename CHANNEL>
void ThreePointCorrMeas<SCALAR,SW_TYPE,CHANNEL>::save_results(const std::string &filename, const alps::mpi::communicator &comm) const {
  std::string dirname = 
    alps::fs::remove_extensions(filename) + "_results";
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