#pragma once

#include <boost/tuple/tuple.hpp>
#include <boost/multi_array.hpp>

#include "../common/wide_scalar.hpp"
#include "../model/operator.hpp"
#include "../model/atomic_model.hpp"

//Measurement of static observable
template<typename SW, typename OBS>
class MeasStaticObs {
 private:
  typedef typename SW::HAM_SCALAR_TYPE SCALAR;

 public:
  MeasStaticObs
      (SW &sw, const operator_container_t &operators); //move the both edges to the same imaginary time in the middle
  void perform_meas(const std::vector<OBS> &obs, std::vector<EXTENDED_COMPLEX> &result) const;
  ~MeasStaticObs(); //restore the sliding window

 private:
  int num_brakets;
  const typename SW::state_t state_bak;
  SW &sw_;
  const operator_container_t &ops_;
};

//Measurement of <O1_i(tau) O2_i(0)> (i=0, ..., N)
// N: the number of correlation functions to be computed
template<typename SW, typename OBS>
class MeasCorrelation {
 private:
  typedef typename SW::HAM_SCALAR_TYPE SCALAR;

 public:
  MeasCorrelation(const std::vector<std::pair<OBS, OBS> > &correlators,
                  int num_tau_points); //move the both edges to the same imaginary time in the middle
  void perform_meas(SW &sw, const operator_container_t &operators, boost::multi_array<EXTENDED_COMPLEX, 2> &result)
      const;

  ~MeasCorrelation(); //restore the sliding window

 private:
  const int num_correlators_;
  const int num_tau_points_, num_win_, right_edge_pos_, left_edge_pos_;
  std::vector<std::pair<int, int> > obs_pos_in_unique_set;
  std::vector<OBS> left_obs_unique_list, right_obs_unique_list;
};
