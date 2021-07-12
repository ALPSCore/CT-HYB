#pragma once

#include <memory>
#include <limits>

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

  SlidingWindowManager() {};

  SlidingWindowManager(const SlidingWindowManager<MODEL> &sw) {
    *this = sw;
  };

  SlidingWindowManager(int n_section, std::shared_ptr<const MODEL> p_model, double beta,
      const operator_container_t &operators = {});

  SlidingWindowManager(std::shared_ptr<const MODEL> p_model, double beta, const std::vector<double> &tau_edges,
      const operator_container_t &operators = {});

  //Initialization
  inline void init_tau_edges(int n_section) {
    tau_edges_.resize(n_section+1);
    tau_edges_[0] = 0;
    for (auto w=1; w<tau_edges_.size()-1; ++w) {
      tau_edges_[w] = (BETA * w) / n_section;
    }
    tau_edges_.back() = BETA;
  }

  void init_stacks();

  bool operator==(const SlidingWindowManager<MODEL> &other) const {
    return 
      this->tau_edges_                   == other.tau_edges_ &&
      this->p_model                      == other.p_model &&
      this->BETA                         == other.BETA &&
      this->num_brakets                  == other.num_brakets &&
      this->norm_cutoff                  == other.norm_cutoff &&
      this->operators                    == other.operators &&
      this->left_states                  == other.left_states &&
      this->right_states                 == other.right_states &&
      this->position_left_edge           == other.position_left_edge &&
      this->position_right_edge          == other.position_right_edge &&
      this->n_section                    == other.n_section &&
      this->direction_move_local_window  == other.direction_move_local_window &&
      this->norm_left_states             == other.norm_left_states &&
      this->norm_right_states            == other.norm_right_states;
  }

  SlidingWindowManager<MODEL>& operator=(const SlidingWindowManager<MODEL> &other) {
    this->tau_edges_                   = other.tau_edges_;
    this->p_model                      = other.p_model;
    this->BETA                         = other.BETA;
    this->num_brakets                  = other.num_brakets;
    this->norm_cutoff                  = other.norm_cutoff;
    this->operators                    = other.operators;
    this->left_states                  = other.left_states;
    this->right_states                 = other.right_states;
    this->position_left_edge           = other.position_left_edge;
    this->position_right_edge          = other.position_right_edge;
    this->n_section                    = other.n_section;
    this->direction_move_local_window  = other.direction_move_local_window;
    this->norm_left_states             = other.norm_left_states;
    this->norm_right_states            = other.norm_right_states;
    return *this;
  }

  //Change window size during MC simulation
  // If new_position_left_edge is not given, it defaults to new_position_left_edge = new_position_right_edge+2.
  void set_uniform_mesh(int n_section_new, 
                      int new_position_right_edge = 0,
                      ITIME_AXIS_LEFT_OR_RIGHT new_direction_move = ITIME_LEFT,
                      int new_position_left_edge = -1);

  //Get and restore the state of the window (size, position, direction of move)
  inline state_t get_state() const {
    return boost::make_tuple(position_left_edge,
                             position_right_edge,
                             direction_move_local_window,
                             n_section);
  }
  void restore_state(state_t state);

  //Getter
  inline int get_num_brakets() const { return num_brakets; };
  inline double get_tau_low() const { return get_tau_edge(position_right_edge); };
  inline double get_tau_high() const { return get_tau_edge(position_left_edge); };

  inline OperatorTime get_op_tau_low(int pos=-1) const {
    if (pos < 0) {
      return OperatorTime(get_tau_edge(position_right_edge), std::numeric_limits<int>::lowest());
    } else {
      return OperatorTime(get_tau_edge(pos), std::numeric_limits<int>::lowest());
    }
  };

  inline OperatorTime get_op_tau_high(int pos=-1) const {
    if (pos < 0) {
      return OperatorTime(get_tau_edge(position_left_edge), std::numeric_limits<int>::max());
    } else {
      return OperatorTime(get_tau_edge(pos), std::numeric_limits<int>::max());
    }
  };
  inline double get_tau_edge(int position) const {return tau_edges_.at(position); }
  inline int get_position_right_edge() const { return position_right_edge; }
  inline int get_position_left_edge() const { return position_left_edge; }
  inline int get_direction_move_local_window() const { return direction_move_local_window; }
  inline double get_tau(int position) const { return tau_edges_[position]; }
  inline int get_n_section() const {return n_section;}
  inline const std::shared_ptr<const MODEL> get_p_model() const { return p_model; }
  inline const operator_container_t& get_operators() const {return operators;}
  inline const double get_beta() const {return BETA;}
  inline const BRAKET_TYPE &get_bra(int bra) const { return left_states[bra].back(); }
  inline const BRAKET_TYPE &get_ket(int ket) const { return right_states[ket].back(); }

  //Manipulation of window
  void move_window_to_next_position();
  void move_backward_edge(ITIME_AXIS_LEFT_OR_RIGHT, int num_move = 1);
  void move_forward_right_edge(int num_move = 1);
  void move_forward_left_edge(int num_move = 1);
  void move_right_edge_to(int pos);
  void move_left_edge_to(int pos);
  void move_edges_to(int left_pos, int right_pos) {
    if (left_pos <= get_position_right_edge()) {
      move_left_edge_to(left_pos);
      move_right_edge_to(right_pos);
    } else {
      move_right_edge_to(right_pos);
      move_left_edge_to(left_pos);
    }
  }
  void move_window_to(ITIME_AXIS_LEFT_OR_RIGHT direction);
  inline void set_direction(ITIME_AXIS_LEFT_OR_RIGHT direction) {
    this->direction_move_local_window = direction;
  }

  //Mnipulation of operators
  std::pair<operator_container_t::iterator,bool> insert(const psi &op) {
    check_true(get_op_tau_low() < op.time() && op.time() < get_op_tau_high());
    return operators.insert(op);
  }

  std::size_t erase(const psi &op) {
    check_true(get_op_tau_low() < op.time() && op.time() < get_op_tau_high());
    return operators.erase(op);
  }

  //Computing trace
  typename ExtendedScalar<typename model_traits<MODEL>::SCALAR_T>::value_type
      compute_trace() const;
  std::pair<bool,
            typename ExtendedScalar<typename model_traits<MODEL>::SCALAR_T>::value_type>
      lazy_eval_trace(EXTENDED_REAL trace_cutoff, std::vector<EXTENDED_REAL> &bound)
      const;
  EXTENDED_REAL compute_trace_bound(std::vector<EXTENDED_REAL> &bound) const;

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

  inline void sanity_check() const;

  // Private member variables
  std::vector<double> tau_edges_;
  std::shared_ptr<const MODEL> p_model;
  double BETA;
  int num_brakets;
  double norm_cutoff;
  operator_container_t operators;
  std::vector<std::vector<BRAKET_TYPE> > left_states, right_states;
  int position_left_edge, position_right_edge, n_section;
  ITIME_AXIS_LEFT_OR_RIGHT direction_move_local_window; //0: left, 1: right
  std::vector<std::vector<EXTENDED_REAL> > norm_left_states, norm_right_states;


};