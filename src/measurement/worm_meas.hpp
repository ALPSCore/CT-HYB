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

/**
 * Record of worm configurations
 **/
template<typename TIME_T, typename VAL_REAL_T>
class WormConfigRecord {
  public:
  WormConfigRecord(int num_time_idx, int num_flavor_idx)
  : num_time_idx_(num_time_idx), num_flavor_idx_(num_flavor_idx),
    taus(num_time_idx), flavors(num_flavor_idx)
   {}

  void add(const Worm &worm, const std::complex<double> &val);

  void save(alps::hdf5::archive &oar, const std::string &path) const;

  void load(alps::hdf5::archive &iar, const std::string &path);

  int nsmpl() const {
    return vals_real.size();
  }

  std::vector<std::vector<TIME_T>> taus;
  std::vector<std::vector<std::uint8_t>> flavors;
  std::vector<VAL_REAL_T> vals_real, vals_imag;

  private:
  const int num_time_idx_, num_flavor_idx_;
};