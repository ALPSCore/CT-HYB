#pragma once

#include <string>

#include "../common/legendre.hpp"
#include "worm_meas.hpp"

template<typename CHANNEL>
class TwoPointCorrMeasGetNameHelper {
  public:
    std::string operator()() const;
};

/**
 * @brief Measurement of two-point correlation function
 */
template <typename SCALAR, typename SW_TYPE, typename CHANNEL>
class TwoPointCorrMeas : public WormMeas<SCALAR,SW_TYPE>
{
public:
  /**
   * Constructor
   */
  TwoPointCorrMeas(alps::random01 *p_rng, double beta, int nflavors, int nl)
    :p_rng_(p_rng), beta_(beta), nflavors_(nflavors), legendre_trans_(0, nl)
  {
    //check_true(is_bosonic(vsample_.begin(), vsample_.end()),
      //"Some of frequencies are not bosonic!");
  }

  void create_alps_observable(
      alps::accumulators::accumulator_set &measurements) const
  {
    //create_observable<std::complex<double>, SimpleRealVectorObservable>(
      //measurements, (get_name()+"_wsample").c_str());
    create_observable<std::complex<double>, SimpleRealVectorObservable>(
      measurements, (get_name()+"_legendre").c_str());
  }

  virtual void measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements);

  virtual void save_results(const std::string &filename) const {
    /*
    alps::mpi::communicator comm;
    if (comm.rank() != 0) {
      return;
    }
    alps::hdf5::archive oar(filename, "a");
    oar[get_name() + "_smpl_freqs"] = vsample_;
    */
  }

private:
  std::string get_name() const {
    return TwoPointCorrMeasGetNameHelper<CHANNEL>()();
  }

  alps::random01 *p_rng_;
  double beta_;
  int nflavors_;
  //std::vector<int> vsample_;
  LegendreTransformer legendre_trans_;
};



void compute_two_point_corr(
    const std::string& name,
    const alps::accumulators::result_set &results,
    int n_flavors,
    double beta,
    double sign,
    double worm_space_rel_vol,
    std::map<std::string,boost::any> &ar,
    bool verbose=false);