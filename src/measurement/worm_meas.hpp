#pragma once

#include <vector>

/**
 * @brief Worm measurement
 * 
 */
template<typename SCALAR, typename SW_TYPE>
class WormMeas {
  public:
  /**
   * Constructor
   */
  WormShiftMeas() {}

  virtual void measure(
    const MonteCarloConfiguration<SCALAR> &mc_config,
    const SlidingWindow &sliding_window,
    alps::accumulators::accumulator_set &measurements) = 0;
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
class WormShiftMeas : public WormMeas {
  public:
    /**
     * Constructor
     */
    WormShiftMeas(
        const std::string& meas_name,
        int shift_target_time_idx,
        const std::vector<double> &taus) :
          shift_target_time_idx_(shift_target_time_idx) {
    }

  
  protected:
    virtual std::vector<double>
      generate_shifted_taus(const MonteCarloConfiguration<SCALAR> &mc_config) = 0;

  private:
    /**
    * @brief Compute MC weights for shifted worms
    * @param mc_config Monte Calor configuration
    * @param sliding_window SlidingWindow obj
    */
    void compute_weight(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SlidingWindow &sliding_window);

    const int shift_target_time_idx_;
};