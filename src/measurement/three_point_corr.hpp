#pragma once

#include <string>

#include "worm_meas.hpp"
#include "alps/numeric/tensors/tensor_base.hpp"

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
  ThreePointCorrMeas(alps::random01 *p_rng, double beta, int nflavors)
    :p_rng_(p_rng), beta_(beta), nflavors_(nflavors), worm_config_record_(3, 4)
  {
  }

  void create_alps_observable(
      alps::accumulators::accumulator_set &measurements) const
  {}

  virtual void measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements);

  virtual void save_results(const std::string &filename, const alps::mpi::communicator &comm) const;

  static void eval_on_smpl_freqs(
    const std::vector<int> &wfs,
    const std::vector<int> &wbs,
    const std::string &datafile,
    const std::string &outputfile
  );

private:
  static
  std::string get_name() {
    return ThreePointCorrMeasGetNameHelper<CHANNEL>()();
  }

  alps::random01 *p_rng_;
  double beta_;
  int nflavors_;

  // Record of worm configuraitons
  WormConfigRecord<double,double> worm_config_record_;
};