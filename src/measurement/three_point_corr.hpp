#pragma once

#include <string>

#include "../common/legendre.hpp"
#include "worm_meas.hpp"

template<typename CHANNEL>
class ThreePointCorrMeasGetNameHelper {
  public:
    std::string operator()() const;
};

/**
 * @brief Measurement of three-point correlation function
 */
template <typename SCALAR, typename SW_TYPE, typename CHANNEL>
class ThreePointCorrMeas : public WormMeas<SCALAR,SW_TYPE>
{
public:
  /**
   * Constructor
   */
  ThreePointCorrMeas(alps::random01 *p_rng, double beta, int nflavors,
    const std::vector<int> &vsample, const std::vector<int> &wsample)
    :p_rng_(p_rng), beta_(beta), nflavors_(nflavors),
     vsample_(vsample), wsample_(wsample)
  {
    check_true(is_fermionic(vsample_.begin(), vsample_.end()), "vsample must be fermionic!");
    check_true(is_bosonic(wsample_.begin(), wsample_.end()), "vsample must be bosonic!");
  }

  void create_alps_observable(
      alps::accumulators::accumulator_set &measurements) const
  {
    create_observable<std::complex<double>, SimpleRealVectorObservable>(
      measurements, get_name().c_str());
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
    oar[get_name() + "/smpl_freqs/0"] = vsample_;
    oar[get_name() + "/smpl_freqs/1"] = wsample_;
  }

private:
  std::string get_name() const {
    return ThreePointCorrMeasGetNameHelper<CHANNEL>()();
  }

  alps::random01 *p_rng_;
  double beta_;
  int nflavors_;
  std::vector<int> vsample_;
  std::vector<int> wsample_;
};