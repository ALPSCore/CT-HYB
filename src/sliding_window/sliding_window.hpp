#pragma once

#include <memory>

#include <boost/tuple/tuple.hpp>
#include <boost/multi_array.hpp>

#include "../common/wide_scalar.hpp"
#include "../model/operator.hpp"
#include "../model/atomic_model.hpp"

enum ITIME_AXIS_LEFT_OR_RIGHT {
  ITIME_LEFT = 0,
  ITIME_RIGHT = 1,
};

//Implementation of sliding window update + lazy trace evaluation
// right edge: tau=0
// left  edge: tau=beta
template<typename MODEL>
class SlidingWindowManager {
 public:
  typedef MODEL IMPURITY_MODEL;
  typedef typename model_traits<MODEL>::SCALAR_T HAM_SCALAR_TYPE;
  typedef typename model_traits<MODEL>::BRAKET_T BRAKET_TYPE;
  typedef typename ExtendedScalar<HAM_SCALAR_TYPE>::value_type EXTENDED_SCALAR;
  typedef typename operator_container_t::iterator op_it_t;
  typedef typename boost::tuple<int, int, ITIME_AXIS_LEFT_OR_RIGHT, int>
      state_t;//pos of left edge, pos of right edge, direction of move, num of windows

  SlidingWindowManager(std::shared_ptr<MODEL> p_model, double beta, int n_window=1, const operator_container_t &operators = {});


  //Initialization
  inline void init_tau_edges(int n_window) {
    tau_edges.resize(2*n_window+1);
    tau_edges[0] = 0;
    for (auto w=1; w<tau_edges.size()-1; ++w) {
      tau_edges[w] = (BETA * w) / (2.0 * n_window);
    }
    tau_edges.back() = BETA;
  }
  //void init_stacks(const operator_container_t &operators = {});

  //Change window size during MC simulation
  // If new_position_left_edge is not given, it defaults to new_position_left_edge = new_position_right_edge+2.
  void set_window_size(int n_window_size, const operator_container_t &operators = {},
                      int new_position_right_edge = 0,
                      ITIME_AXIS_LEFT_OR_RIGHT new_direction_move = ITIME_LEFT,
                      int new_position_left_edge = -1);

  //Get and restore the state of the window (size, position, direction of move)
  inline state_t get_state() const {
    return boost::make_tuple(position_left_edge,
                             position_right_edge,
                             direction_move_local_window,
                             n_window);
  }
  void restore_state(const operator_container_t &ops, state_t state);

  //Getter
  inline int get_num_brakets() const { return num_brakets; };
  inline double get_tau_low() const { return get_tau_edge(position_right_edge); };
  inline double get_tau_high() const { return get_tau_edge(position_left_edge); };
  inline double get_tau_edge(int position) const {return tau_edges.at(position); }
  inline int get_n_window() const { return n_window; };
  inline int get_position_right_edge() const { return position_right_edge; }
  inline int get_position_left_edge() const { return position_left_edge; }
  inline int get_direction_move_local_window() const { return direction_move_local_window; }
  inline const std::shared_ptr<const MODEL> get_p_model() const { return p_model; }
  inline const BRAKET_TYPE &get_bra(int bra) const { return left_states[bra].back(); }
  inline const BRAKET_TYPE &get_ket(int ket) const { return right_states[ket].back(); }

  //Manipulation of window
  void move_window_to_next_position(const operator_container_t &operators);
  void move_backward_edge(ITIME_AXIS_LEFT_OR_RIGHT, int num_move = 1);
  void move_forward_right_edge(const operator_container_t &operators, int num_move = 1);
  void move_forward_left_edge(const operator_container_t &operators, int num_move = 1);
  void move_right_edge_to(const operator_container_t &operators, int pos);
  void move_left_edge_to(const operator_container_t &operators, int pos);
  void move_window_to(const operator_container_t &operators, ITIME_AXIS_LEFT_OR_RIGHT direction);

  //Computing trace
  typename ExtendedScalar<typename model_traits<MODEL>::SCALAR_T>::value_type
      compute_trace(const operator_container_t &ops) const;
  std::pair<bool,
            typename ExtendedScalar<typename model_traits<MODEL>::SCALAR_T>::value_type>
      lazy_eval_trace(const operator_container_t &ops, EXTENDED_REAL trace_cutoff, std::vector<EXTENDED_REAL> &bound)
      const;
  EXTENDED_REAL compute_trace_bound(const operator_container_t &ops, std::vector<EXTENDED_REAL> &bound) const;

  //static function for imaginary-time evolution of a bra or a ket
  static void evolve_bra
      (const MODEL &model, BRAKET_TYPE &bra, std::pair<op_it_t, op_it_t> ops_range, double tau_old, double tau_new);
  static void evolve_ket
      (const MODEL &model, BRAKET_TYPE &ket, std::pair<op_it_t, op_it_t> ops_range, double tau_old, double tau_new);

 private:
  inline int depth_left_states() const { return left_states[0].size(); }
  inline int depth_right_states() const { return right_states[0].size(); }
  void pop_back_bra(int num_pop_back = 1);
  void pop_back_ket(int num_pop_back = 1);
  inline double compute_exp(int sector, double tau) const {
    const double limit = std::log(std::numeric_limits<double>::min()) / 2;
    const double prod = -tau * p_model->min_energy(sector);
    if (prod < limit) {
      return 0.0;
    } else {
      return std::exp(-tau * p_model->min_energy(sector));
    }
  }
  inline bool is_braket_invalid(int braket) const {
    return right_states[braket].back().invalid() || left_states[braket].back().invalid();
  }
  inline typename ExtendedScalar<typename model_traits<MODEL>::SCALAR_T>::value_type
      compute_trace_braket(int braket, std::pair<op_it_t, op_it_t> ops_range, double tau_left, double tau_right) const;

  // Private member variables
  std::vector<double> tau_edges;
  const std::shared_ptr<MODEL> p_model;
  const double BETA;
  const int num_brakets;
  const double norm_cutoff;

  std::vector<std::vector<BRAKET_TYPE> > left_states, right_states;
  //bra and ket, respectively
  int position_left_edge, position_right_edge, n_window;
  ITIME_AXIS_LEFT_OR_RIGHT direction_move_local_window; //0: left, 1: right

  //for lazy evalulation of trace using spectral norm
  std::vector<std::vector<EXTENDED_REAL> > norm_left_states, norm_right_states;

  inline void sanity_check() const;
};