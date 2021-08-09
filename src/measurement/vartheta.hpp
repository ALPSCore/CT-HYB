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
class VarThetaMeas : public WormMeas<SCALAR,SW_TYPE>
{
public:
  /**
   * Constructor
   */
  VarThetaMeas(alps::random01 *p_rng, double beta, int nflavors, const std::vector<int>& vsample)
    :p_rng_(p_rng), beta_(beta), nflavors_(nflavors), vsample_(vsample)
  {
    check_true(is_fermionic(vsample_.begin(), vsample_.end()), "Some of frequencies are not fermionic!");
  }

  void create_alps_observable(
      alps::accumulators::accumulator_set &measurements) const
  {
    create_observable<std::complex<double>, SimpleRealVectorObservable>(measurements, "vartheta");
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
    oar["vartheta_smpl_freqs"] = vsample_;
  }

private:
  alps::random01 *p_rng_;
  double beta_;
  int nflavors_;
  std::vector<int> vsample_;
};

void compute_vartheta(
    const alps::accumulators::result_set &results,
    int n_flavors,
    double beta,
    double sign,
    double worm_space_rel_vol,
    std::map<std::string,boost::any> &ar,
    bool verbose=false);

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

void compute_vartheta_legendre(
    const alps::accumulators::result_set &results,
    int n_flavors,
    double beta,
    double sign,
    double worm_space_rel_vol,
    std::map<std::string,boost::any> &ar,
    bool verbose=false);