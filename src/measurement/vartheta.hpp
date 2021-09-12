#pragma once

#include "worm_meas.hpp"
#include "../common/legendre.hpp"

/**
 * @brief Measurement of \vartheta_ab = -<T_tau q_a(\tau) q_b(0)>.
 * 
 * Multiple MC samples are generated on the fly 
 * by shifting the worm in time.
 */
template <typename SCALAR, typename SW_TYPE>
class VarThetaLegendreMeas : public WormMeas<SCALAR,SW_TYPE>
{
public:
  /**
   * Constructor
   */
  VarThetaLegendreMeas(alps::random01 *p_rng, double beta, int nflavors,
    int nl): p_rng_(p_rng), beta_(beta), nflavors_(nflavors),
      legendre_trans_(0, nl)
  {}

  void create_alps_observable(
      alps::accumulators::accumulator_set &measurements) const
  {
    create_observable<std::complex<double>, SimpleRealVectorObservable>(
      measurements, "vartheta_legendre");
  }

  virtual void measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements);

  virtual void save_results(const std::string &filename) const { }

private:
  alps::random01 *p_rng_;
  double beta_;
  int nflavors_;
  LegendreTransformer legendre_trans_;
};