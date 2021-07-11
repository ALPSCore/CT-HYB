#pragma once
 
#include <vector>
#include <array>
#include <alps/accumulators.hpp>
#include "../accumulator.hpp"
#include "../moves/worm.hpp"

/**
 * @brief Base class for worm measurement
 * 
 */
template<typename SCALAR, typename SW_TYPE>
class WormMeasBase {
  public:
  /**
   * Constructor
   */
  WormShiftMeas() {}

  virtual void create_alps_observable(
    alps::accumulators::accumulator_set &measurements) const {}

  virtual void measure(
    const MonteCarloConfiguration<SCALAR> &mc_config,
    const SlidingWindow &sliding_window,
    alps::accumulators::accumulator_set &measurements) = 0;
};


/**
 * Measure equal-time Green's function <c^dagger c , ..., c^dagger c>
 * Rank = 1: single-particle Green's function
 * Rank = 2: two-particle Green's function
 *
 */
template<unsigned int Rank>
class EqualTimeGWorm: public Worm, private boost::equality_comparable<EqualTimeGWorm<Rank> > {
 public:
  EqualTimeGWorm() {
    time_index_.push_back(0);
  }

  virtual boost::shared_ptr<Worm> clone() const {
    return boost::shared_ptr<Worm>(new EqualTimeGWorm<Rank>(*this));
  }

  virtual int num_operators() const { return 2 * Rank; };

  virtual std::vector<psi> get_operators() const;//implemented in worm.ipp

  virtual int num_independent_times() const { return 1; }

  virtual double get_time(int index) const {
    assert(index == 0);
    return time_;
  }

  virtual void set_time(int index, double new_time) {
    assert(index == 0);
    time_ = new_time;
  }

  virtual int num_independent_flavors() const { return 2 * Rank; }

  virtual int get_flavor(int index) const {
    assert(index >= 0 && index < 2 * Rank);
    return flavors_[index];
  }

  virtual void set_flavor(int index, int new_flavor) {
    assert(index >= 0 && index < 2 * Rank);
    flavors_[index] = new_flavor;
  }

  virtual const std::vector<int> &get_time_index(int flavor_index) const {
    assert(flavor_index >= 0 && flavor_index < 2 * Rank);
    return time_index_;
  }

  virtual bool operator==(const EqualTimeGWorm<Rank> &other_worm) const {
    return (time_ == other_worm.time_ && flavors_ == other_worm.flavors_);
  }

  ConfigSpace get_config_space() const {
    return Unknown;
  }

 private:
  double time_;
  std::array<int, 2 * Rank> flavors_;
  std::vector<int> time_index_;
};


/**
 * @brief Measurement by worm shift in tau 
 * 
 * Select one time index of a worm and shift it in tau 
 * to generate multiple MC samples.
 * These MC samples and the original MC sample 
 * will be measured.
 */
template<typename SCALAR, typename SW_TYPE>
class EqualTimeG1Meas : public WormMeasBase {
  public:
  /**
   * Constructor
   */
  EqualTimeG1Meas(als::random01 &rng, int num_ins) :
    p_rng_(&rng), num_ins_(num_ins) {
  }
    
  void create_alps_observable(
    alps::accumulators::accumulator_set &measurements) const {
      create_observable<std::complex<double>, SimpleRealVectorObservable>(measurements, "Equal_time_G1".c_str());
  }
  
  protected:
  /**
  * @brief Compute MC weights for shifted worms
  * @param mc_config Monte Calor configuration
  * @param sliding_window SlidingWindow obj
  * @param taus_ins tau points where worms are placed at
  * @param weights  Relative weights (local trace + sign) of new worm configurations
  */
  //void compute_weight(
    //const MonteCarloConfiguration<SCALAR> &mc_config,
    //const SlidingWindow &sliding_window,
    //const std::vector<double> &taus_ins,
    //std::<double> weights
    //);

  virtual void measure(
    const MonteCarloConfiguration<SCALAR> &mc_config,
    const SlidingWindow &sliding_window,
    alps::accumulators::accumulator_set &measurements) {
      MonteCarloConfiguration<SCALAR> mc_config_wrk(mc_config);
      SlidingWindow sw_wrk(sliding_window);

      /*
      boost::shared_ptr<Worm> p_worm_org;
      if (mc_config.p_worm) {
        p_worm_org = mc_config.p_worm.clone();
      }


      if (p_worm_org) {
        mc_config.set_worm(*p_worm_org);
      }
      */
    }

  private:
  alps::random01 *p_rng;
  int num_ins_;
};