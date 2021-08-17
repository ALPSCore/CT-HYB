#pragma once

#include <string>

#include "worm_meas.hpp"

/**
 * @brief Measurement of correlation function `h'
 */
template <typename SCALAR, typename SW_TYPE>
class HCorrMeas : public WormMeas<SCALAR,SW_TYPE>
{
public:
  /**
   * Constructor
   */
  HCorrMeas(alps::random01 *p_rng, double beta, int nflavors,
    const std::vector<int> &v1,
    const std::vector<int> &v2,
    const std::vector<int> &v3,
    const std::vector<int> &v4)
    :p_rng_(p_rng), beta_(beta), nflavors_(nflavors),
     v1_(v1), v2_(v2), v3_(v3), v4_(v4)
  {
    check_true(is_fermionic(v1_.begin(), v1_.end()), "v1 must be fermionic!");
    check_true(is_fermionic(v2_.begin(), v2_.end()), "v2 must be fermionic!");
    check_true(is_fermionic(v3_.begin(), v3_.end()), "v3 must be fermionic!");
    check_true(is_fermionic(v4_.begin(), v4_.end()), "v4 must be fermionic!");
  }

  void create_alps_observable(
      alps::accumulators::accumulator_set &measurements) const
  {
    create_observable<std::complex<double>, SimpleRealVectorObservable>(
      measurements, "h_corr");
  }

  virtual void measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements);

  virtual void save_results(const std::string &filename, const alps::mpi::communicator &comm) const {
    if (comm.rank() != 0) {
      return;
    }
    alps::hdf5::archive oar(filename, "a");
    oar["h_corr/smpl_freqs/0"] = v1_;
    oar["h_corr/smpl_freqs/1"] = v2_;
    oar["h_corr/smpl_freqs/2"] = v3_;
    oar["h_corr/smpl_freqs/3"] = v4_;
  }

private:
  alps::random01 *p_rng_;
  double beta_;
  int nflavors_;
  std::vector<int> v1_, v2_, v3_, v4_;
};