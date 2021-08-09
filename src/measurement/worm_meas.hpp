#pragma once

#include <vector>
#include <algorithm>
#include <map>
#include <array>
#include <boost/any.hpp>

#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/hdf5.hpp>

#include "../accumulator.hpp"
#include "../moves/mc_config.hpp"
#include "../moves/worm.hpp"
#include "../moves/operator_util.hpp"
#include "../sliding_window/sliding_window.hpp"


template <typename SW_TYPE>
typename SW_TYPE::EXTENDED_SCALAR
compute_trace_worm_impl(SW_TYPE &sw, const std::vector<psi> &ops) {
  for (const auto& op: ops) {
    sw.insert(op);
  }
  auto trace = sw.compute_trace();
  for (const auto& op: ops) {
    sw.erase(op);
  }
  return trace;
}

/**
 * @brief Base class for worm measurement
 * 
 */
template <typename SCALAR, typename SW_TYPE>
class WormMeas
{
public:
  /**
   * Constructor
   */
  WormMeas() {}

  virtual void create_alps_observable(
      alps::accumulators::accumulator_set &measurements) const {}

  virtual void measure(
    const MonteCarloConfiguration<SCALAR> &mc_config,
    const SW_TYPE &sliding_window,
    alps::accumulators::accumulator_set &measurements) = 0;

  virtual void save_results(const std::string &filename, const alps::mpi::communicator &comm) const {};
};


/**
 * @brief Measurement of equal-time G1
 * 
 * Multiple MC samples are generated on the fly 
 * by shifting the worm in time.
 */
template <typename SCALAR, typename SW_TYPE>
class EqualTimeG1Meas : public WormMeas<SCALAR,SW_TYPE>
{
public:
  /**
   * Constructor
   */
  EqualTimeG1Meas(alps::random01 *p_rng, double beta, int nflavors, int num_meas)
    :p_rng_(p_rng), beta_(beta), nflavors_(nflavors), num_meas_(num_meas)
  {
  }

  void create_alps_observable(
      alps::accumulators::accumulator_set &measurements) const
  {
    create_observable<std::complex<double>, SimpleRealVectorObservable>(measurements, "Equal_time_G1");
  }

  virtual void measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements);

private:
  alps::random01 *p_rng_;
  double beta_;
  int nflavors_, num_meas_;
};


void compute_equal_time_G1(
    const alps::accumulators::result_set &results,
    int n_flavors,
    double beta,
    double sign,
    double G1_space_vol_rat,
    std::map<std::string,boost::any> &ar,
    bool verbose = false);

std::vector<int> read_fermionic_matsubara_points(const std::string& file);


inline std::pair<double,double>
fermionic_sign_time_ordering(double tau, double beta) {
  double tau_sign = 1.0;
  while (tau < 0) {
    tau += beta;
    tau_sign *= -1.0;
  }
  while (tau > beta) {
    tau -= beta;
    tau_sign *= -1.0;
  }
  return {tau, tau_sign};
}