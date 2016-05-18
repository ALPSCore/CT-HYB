#pragma once

#include <boost/tuple/tuple.hpp>
#include <boost/multi_array.hpp>

#include "operator.hpp"
#include "model.hpp"

enum ITIME_AXIS_LEFT_OR_RIGHT {
  ITIME_LEFT = 0,
  ITIME_RIGHT = 1,
};

//Implementation of sliding window update + lazy trace evaluation
template<typename MODEL>
class SlidingWindowManager
{
public:
  typedef MODEL IMPURITY_MODEL;
  typedef typename model_traits<MODEL>::SCALAR_T HAM_SCALAR_TYPE;
  typedef typename model_traits<MODEL>::BRAKET_T  BRAKET_TYPE;//class Braket is defined in model.hpp
  typedef typename operator_container_t::iterator op_it_t;
  typedef typename boost::tuple<int,int,ITIME_AXIS_LEFT_OR_RIGHT> state_t;//pos of left edge, pos of right edge, direction of move

  SlidingWindowManager(MODEL* p_model, double beta);

  //Initialization
  void init_stacks(int n_window_size, const operator_container_t& operators);

  //Change window size during MC simulation
  void set_window_size(int n_window_size, const operator_container_t& operators, int new_position_right_edge=0,
                       ITIME_AXIS_LEFT_OR_RIGHT new_direction_move=ITIME_LEFT);

  //Get and restore the state of the window (size, position, direction of move)
  inline state_t get_state() const {return boost::make_tuple(position_left_edge,position_right_edge,direction_move_local_window);}
  void restore_state(const operator_container_t& ops, state_t state);

  //Getter
  inline int get_num_brakets() const {return num_brakets;};
  inline double get_tau_low() const {return get_tau_edge(position_right_edge);};
  inline double get_tau_high() const  {return get_tau_edge(position_left_edge);};
  inline double get_tau_edge(int position) const { return (BETA*position)/(2.0*n_window);}
  inline int get_n_window() const {return n_window;};
  inline int get_position_right_edge() const {return position_right_edge;}
  inline int get_position_left_edge() const {return position_left_edge;}
  inline int get_direction_move_local_window() const {return direction_move_local_window;}
  inline const MODEL* get_p_model() const {return p_model;}
  inline const BRAKET_TYPE& get_bra(int bra) const {return left_states[bra].back();}
  inline const BRAKET_TYPE& get_ket(int ket) const {return right_states[ket].back();}

  //Manipulation of window
  void move_window_to_next_position(const operator_container_t& operators);
  void move_backward_edge(ITIME_AXIS_LEFT_OR_RIGHT , int num_move=1);
  void move_forward_right_edge(const operator_container_t& operators, int num_move=1);
  void move_forward_left_edge(const operator_container_t& operators, int num_move=1);
  void move_right_edge_to(const operator_container_t& operators, int pos);
  void move_left_edge_to(const operator_container_t& operators, int pos);
  void move_window_to(const operator_container_t& operators, ITIME_AXIS_LEFT_OR_RIGHT direction);

  //Computing trace
  typename model_traits<MODEL>::SCALAR_T compute_trace(const operator_container_t& ops) const;
  std::pair<bool,typename model_traits<MODEL>::SCALAR_T> lazy_eval_trace(const operator_container_t& ops, double trace_cutoff,
                                                                         std::vector<double>& bound) const;
  double compute_trace_bound(const operator_container_t& ops, std::vector<double>& bound) const;

  //static function for imaginary-time evolution of a bra or a ket
  static void evolve_bra(const MODEL& model, BRAKET_TYPE& bra, std::pair<op_it_t,op_it_t> ops_range, double tau_old, double tau_new);
  static void evolve_ket(const MODEL& model, BRAKET_TYPE& ket, std::pair<op_it_t,op_it_t> ops_range, double tau_old, double tau_new);

private:
  const MODEL* const p_model;
  const double BETA;
  const int num_brakets;
  const double norm_cutoff;

  inline int depth_left_states() const {return left_states[0].size();}
  inline int depth_right_states() const {return right_states[0].size();}
  void pop_back_bra(int num_pop_back=1);
  void pop_back_ket(int num_pop_back=1);
  inline double compute_exp(int sector, double tau) const {
      return std::exp(-tau*p_model->min_energy(sector));
  }
  inline bool is_braket_invalid(int braket) const {
      return right_states[braket].back().invalid() || left_states[braket].back().invalid();
  }
  inline typename model_traits<MODEL>::SCALAR_T compute_trace_braket(int braket,
                                                                     std::pair<op_it_t,op_it_t> ops_range, double tau_left, double tau_right) const;

  std::vector<std::vector<BRAKET_TYPE> > left_states, right_states;//bra and ket, respectively
  int position_left_edge, position_right_edge, n_window;
  ITIME_AXIS_LEFT_OR_RIGHT direction_move_local_window; //0: left, 1: right

  //for lazy evalulation of trace using spectral norm
  std::vector<std::vector<double> > norm_left_states, norm_right_states;

  inline void sanity_check() const;
};

//Measurement of static observable
template<typename SW, typename OBS>
class MeasStaticObs
{
private:
  typedef typename SW::HAM_SCALAR_TYPE SCALAR;

public:
  MeasStaticObs(SW& sw, const operator_container_t& operators); //move the both edges to the same imaginary time in the middle
  void perform_meas(const std::vector<OBS>& obs, std::vector<SCALAR>& result) const;
  ~MeasStaticObs(); //restore the sliding window

private:
  int num_brakets;
  const typename SW::state_t state_bak;
  SW& sw_;
  const operator_container_t& ops_;
};

//Measurement of <O1_i(tau) O2_i(0)> (i=0, ..., N)
// N: the number of correlation functions to be computed
template<typename SW, typename OBS>
class MeasCorrelation {
private:
  typedef typename SW::HAM_SCALAR_TYPE SCALAR;

public:
  MeasCorrelation(const std::vector<std::pair<OBS,OBS> >& correlators, int num_tau_points); //move the both edges to the same imaginary time in the middle
  void perform_meas(SW& sw, const operator_container_t &operators, boost::multi_array<std::complex<double>,2>& result) const;

  ~MeasCorrelation(); //restore the sliding window

private:
  const int num_correlators_;
  const int num_tau_points_, right_edge_pos_, left_edge_pos_;
  std::vector<std::pair<int,int> > obs_pos_in_unique_set;
  std::vector<OBS> left_obs_unique_list, right_obs_unique_list;
};
