#include "g1.hpp"

#include <alps/utilities/fs/get_dirname.hpp>
#include <alps/utilities/fs/get_basename.hpp>
#include <alps/utilities/fs/remove_extensions.hpp>

template <typename SCALAR, typename SW_TYPE>
void 
G1Meas<SCALAR,SW_TYPE>::measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements)
{
  auto expix = [](double x) {return std::complex<double>(std::cos(x), std::sin(x));};
  auto temperature = 1/beta_;
  for (auto ifreq=0; ifreq<vsample_.size(); ++ifreq) {
    auto tau = mc_config.p_worm->get_time(0) - mc_config.p_worm->get_time(1);
    g1_data_(ifreq,
        mc_config.p_worm->get_flavor(0),
        mc_config.p_worm->get_flavor(1)) += -expix(M_PI * temperature * (vsample_[ifreq]*tau)) * mc_config.sign;
  }
  ++ num_data_;
};

template <typename SCALAR, typename SW_TYPE>
void G1Meas<SCALAR,SW_TYPE>::save_results(const std::string &filename, const alps::mpi::communicator &comm) const {
  alps::numerics::tensor<double,3> data_real(vsample_.size(), nflavors_, nflavors_);
  alps::numerics::tensor<double,3> data_imag(vsample_.size(), nflavors_, nflavors_);
  for (auto i=0; i<vsample_.size(); ++i) {
    for (auto j=0; j<nflavors_; ++j) {
      for (auto k=0; k<nflavors_; ++k) {
        data_real(i,j,k) = g1_data_(i,j,k).real()/num_data_;
        data_imag(i,j,k) = g1_data_(i,j,k).imag()/num_data_;
      }
    }
  }
  for (int r=0; r<comm.size(); ++r) {
    if (r == comm.rank()) {
      alps::hdf5::archive oar(filename, "a");
      if (r == 0) {
        oar["/giv/num_data_sets"] = comm.size();
        oar["/giv/vsample"] = vsample_;
      }
      oar["/giv/"+std::to_string(r)+"/real"] = data_real;
      oar["/giv/"+std::to_string(r)+"/imag"] = data_imag;
    }
    comm.barrier();
  }
}
