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
  HCorrMeas(alps::random01 *p_rng, double beta, int nflavors)
    :p_rng_(p_rng), beta_(beta), nflavors_(nflavors), worm_config_record_(4, 4)
  {}

  void create_alps_observable(
      alps::accumulators::accumulator_set &measurements) const
  {
  }

  virtual void measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements);

  virtual void save_results(const std::string &filename, const alps::mpi::communicator &comm) const {
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

private:
  alps::random01 *p_rng_;
  double beta_;
  int nflavors_;
  std::vector<int> v1_, v2_, v3_, v4_;
  int nsmpl_;

  static
  std::string get_name() {
    return "h_corr";
  }

  // Record of worm configuraitons
  WormConfigRecord<double,double> worm_config_record_;
};