#pragma once

#include "worm_meas.hpp"
#include "../common/legendre.hpp"

/**
 * @brief Measurement of G_ab = -<T_tau c_a(\tau) c^\dagger_b(0)>.
 */
template <typename SCALAR, typename SW_TYPE>
class G1Meas : public WormMeas<SCALAR,SW_TYPE>
{
public:
  /**
   * Constructor
   */
  G1Meas(alps::random01 *p_rng, double beta, int nflavors, const std::vector<int>& vsample)
    :p_rng_(p_rng), beta_(beta), nflavors_(nflavors), vsample_(vsample), num_data_(0), g1_data_(vsample_.size(), nflavors, nflavors)
  {
    check_true(is_fermionic(vsample_.begin(), vsample_.end()), "Some of frequencies are not fermionic!");
    g1_data_.set_number(0.0);
  }

  void create_alps_observable(
      alps::accumulators::accumulator_set &measurements) const
  {}

  virtual void measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements);

  virtual void save_results(const std::string &filename, const alps::mpi::communicator &comm) const;

private:
  alps::random01 *p_rng_;
  double beta_;
  int nflavors_;
  std::vector<int> vsample_;
  int num_data_;
  alps::numerics::tensor<std::complex<double>,3> g1_data_;
};