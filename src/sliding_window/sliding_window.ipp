#include "sliding_window.hpp"


template<typename MODEL>
SlidingWindowManager<MODEL>::SlidingWindowManager(std::shared_ptr<const MODEL> p_model_,
  double beta, const std::vector<double> &tau_edges, const operator_container_t &operators)
    : p_model(p_model_),
      BETA(beta),
      n_section(tau_edges.size()-1),
      tau_edges_(tau_edges),
      num_brakets(p_model->num_brakets()),
      norm_cutoff(std::sqrt(std::numeric_limits<double>::min())),
      operators()
       {
  check_true(tau_edges_[0]==0.0 && tau_edges_.back()==beta, "tau_edges is invalid!");
  for (auto& op: operators) {
    this->operators.insert(op);// Duplicate keys are checked!
  }
  for (auto i=0; i<tau_edges_.size()-1; ++i) {
    check_true(tau_edges_[i] < tau_edges_[i+1], "tau_edges must be in strictly increasing order!");
  }
  position_left_edge = n_section;
  position_right_edge = 0;

  init_stacks();
  sanity_check();
}

template<typename MODEL>
void SlidingWindowManager<MODEL>::init_stacks() {
  left_states.resize(num_brakets);
  right_states.resize(num_brakets);
  norm_left_states.resize(num_brakets);//for bra
  norm_right_states.resize(num_brakets);//for ket
  for (int braket = 0; braket < num_brakets; ++braket) {
    left_states[braket].resize(0);
    right_states[braket].resize(0);
    left_states[braket].push_back(
        p_model->get_outer_bra(braket)
    );
    right_states[braket].push_back(
        p_model->get_outer_ket(braket)
    );
    norm_left_states[braket].resize(0);
    norm_right_states[braket].resize(0);
    norm_left_states[braket].push_back(left_states[braket].back().compute_spectral_norm());
    norm_right_states[braket].push_back(right_states[braket].back().compute_spectral_norm());
  }
}

template<typename MODEL>
SlidingWindowManager<MODEL>::SlidingWindowManager(int n_section,
  std::shared_ptr<const MODEL> p_model_, double beta, const operator_container_t &operators)
    : p_model(p_model_),
      BETA(beta),
      n_section(n_section),
      num_brakets(p_model->num_brakets()),
      norm_cutoff(std::sqrt(std::numeric_limits<double>::min())),
      operators(operators)
       {

  // Generate meshes
  init_tau_edges(n_section);

  position_left_edge = n_section;
  position_right_edge = 0;

  init_stacks();

  sanity_check();
}

template<typename MODEL>
void SlidingWindowManager<MODEL>::set_mesh(const std::vector<double> &tau_edges,
                                                  int new_position_right_edge,
                                                  ITIME_AXIS_LEFT_OR_RIGHT new_direction_move,
                                                  int new_position_left_edge
                                                  ) {
  check_true(tau_edges[0] == 0.0);
  check_true(tau_edges.back() == BETA);
  for (auto t=0; t<tau_edges.size()-1; ++t) {
    check_true(tau_edges[t] < tau_edges[t+1], "tau_edges must be in strictly ascending order!");
  }

  if (new_position_left_edge < 0) {
    new_position_left_edge = new_position_right_edge+2;
  }

  //reset
  while (depth_right_states() > 1) {
    move_backward_edge(ITIME_RIGHT);
  }
  while (depth_left_states() > 1) {
    move_backward_edge(ITIME_LEFT);
  }

  tau_edges_ = tau_edges;
  n_section = tau_edges_.size()-1;

  position_right_edge = 0;
  position_left_edge = n_section;

  for (int i = 0; i < new_position_right_edge; ++i) {
    move_forward_right_edge();
  }
  for (int i = 0; i < n_section - new_position_left_edge; ++i) {
    move_forward_left_edge();
  }
  check_true(position_left_edge == new_position_left_edge);
  check_true(position_right_edge == new_position_right_edge);

  direction_move_local_window = new_direction_move;

  sanity_check();

}

template<typename MODEL>
void SlidingWindowManager<MODEL>::set_uniform_mesh(int n_section_new,
                                                  int new_position_right_edge,
                                                  ITIME_AXIS_LEFT_OR_RIGHT new_direction_move,
                                                  int new_position_left_edge
                                                  ) {
  std::vector<double> tau_edges(n_section_new+1);
  tau_edges[0] = 0;
  for (auto w=1; w<tau_edges.size()-1; ++w) {
    tau_edges[w] = (BETA * w) / n_section_new;
  }
  tau_edges.back() = BETA;

  set_mesh(tau_edges, new_position_right_edge, new_direction_move, new_position_left_edge);
}

template<typename MODEL>
void
SlidingWindowManager<MODEL>::move_backward_edge(ITIME_AXIS_LEFT_OR_RIGHT which_edge, int num_move) {
  sanity_check();
  if (which_edge == ITIME_RIGHT) {
    if (static_cast<int>(position_right_edge) - num_move < 0) {
      throw std::runtime_error("Out of range in move_backward_edge");
    }
    pop_back_ket(num_move);
    position_right_edge -= num_move;
  } else if (which_edge == ITIME_LEFT) {
    if (position_left_edge + num_move > n_section) {
      throw std::runtime_error("Out of range in move_backward_edge");
    }
    pop_back_bra(num_move);
    position_left_edge += num_move;
  } else {
    throw std::runtime_error("Unknown edge");
  }
  sanity_check();
}

//small-beta side
template<typename MODEL>
void
SlidingWindowManager<MODEL>::move_forward_right_edge(int num_move) {
  namespace bll = boost::lambda;
  sanity_check();

  for (int move = 0; move < num_move; ++move) {
    //range check
    check_true(position_right_edge >= 0);
    check_true(position_right_edge <= n_section);
    OperatorTime tau_edge_old = get_op_tau_low(position_right_edge);
    OperatorTime tau_edge_new = get_op_tau_low(position_right_edge + 1);
    auto ops_range = operators.range(tau_edge_old <= bll::_1, bll::_1 < tau_edge_new);
    EXTENDED_REAL max_norm = -1;
    for (int i_braket = 0; i_braket < num_brakets; ++i_braket) {
      right_states[i_braket].push_back(right_states[i_braket].back());
      evolve_ket(*p_model, right_states[i_braket].back(), ops_range, tau_edge_old.time(), tau_edge_new.time());
      norm_right_states[i_braket].push_back(right_states[i_braket].back().compute_spectral_norm());
      if (max_norm < norm_right_states[i_braket].back()) {
        max_norm = norm_right_states[i_braket].back();
      }
    }
    for (int i_braket = 0; i_braket < num_brakets; ++i_braket) {
      if (norm_right_states[i_braket].back() <
          (EXTENDED_REAL(max_norm)) * (EXTENDED_REAL(1E-100))
          ) {
        norm_right_states[i_braket].back() = 0.0;
        right_states[i_braket].back().set_invalid();
      }
      assert(norm_right_states[i_braket].back() >= 0.0);
    }
    ++position_right_edge;
    sanity_check();
  }
}


//large-beta side
template<typename MODEL>
void
SlidingWindowManager<MODEL>::move_forward_left_edge(int num_move) {
  namespace bll = boost::lambda;

  for (int move = 0; move < num_move; ++move) {
    //range check
    check_true(position_left_edge >= 0 && position_left_edge <= n_section);

    sanity_check();

    OperatorTime tau_edge_old = get_op_tau_high(position_left_edge);
    OperatorTime tau_edge_new = get_op_tau_high(position_left_edge - 1);
    const std::pair<op_it_t, op_it_t> ops_range = operators.range(tau_edge_new < bll::_1,
                                                                      bll::_1 <= tau_edge_old);

    EXTENDED_REAL max_norm = -1;
    for (int i_braket = 0; i_braket < num_brakets; ++i_braket) {
      left_states[i_braket].push_back(left_states[i_braket].back());
      evolve_bra(*p_model, left_states[i_braket].back(), ops_range, tau_edge_old.time(), tau_edge_new.time());
      norm_left_states[i_braket].push_back(left_states[i_braket].back().compute_spectral_norm());
      if (max_norm < norm_left_states[i_braket].back()) {
        max_norm = norm_left_states[i_braket].back();
      }
    }
    for (int i_braket = 0; i_braket < num_brakets; ++i_braket) {
      if (norm_left_states[i_braket].back() < max_norm * 1E-100) {
        norm_left_states[i_braket].back() = 0.0;
        left_states[i_braket].back().set_invalid();
      }
    }
    --position_left_edge;
  }

  sanity_check();
}

template<typename MODEL>
void
SlidingWindowManager<MODEL>::move_right_edge_to(int pos) {
  assert(pos >= 0 && pos <= n_section);
  if (get_position_right_edge() > pos) {
    move_backward_edge(ITIME_RIGHT, get_position_right_edge() - pos);
  } else if (get_position_right_edge() < pos) {
    move_forward_right_edge(pos - get_position_right_edge());
  }
}

template<typename MODEL>
void
SlidingWindowManager<MODEL>::move_left_edge_to(int pos) {
  assert(pos >= 0 && pos <= n_section);
  const int current_pos = get_position_left_edge();
  if (current_pos > pos) {
    move_forward_left_edge(current_pos - pos);
  } else if (current_pos < pos) {
    move_backward_edge(ITIME_LEFT, pos - current_pos);
  }
}


template<typename MODEL>
typename ExtendedScalar<typename model_traits<MODEL>::SCALAR_T>::value_type
SlidingWindowManager<MODEL>::compute_trace() const {
  namespace bll = boost::lambda;

  sanity_check();

  EXTENDED_SCALAR trace = 0.0;

  OperatorTime tau_right = get_op_tau_low(position_right_edge);
  OperatorTime tau_left = get_op_tau_high(position_left_edge);
  std::pair<op_it_t, op_it_t> ops_range = operators.range(tau_right <= bll::_1, bll::_1 <= tau_left);

  for (int i_braket = 0; i_braket < num_brakets; ++i_braket) {
    if (is_braket_invalid(i_braket)) {
      continue;
    }
    BRAKET_TYPE ket = right_states[i_braket].back();

    evolve_ket(*p_model, ket, ops_range, tau_right.time(), tau_left.time());
    if (left_states[i_braket].back().sector() == ket.sector()) {
      const EXTENDED_SCALAR trace_braket = p_model->product(left_states[i_braket].back(), ket);
      assert(!my_isnan(trace_braket));
      trace += trace_braket;
    }
  }
  return trace;
}

template<typename MODEL>
typename ExtendedScalar<typename model_traits<MODEL>::SCALAR_T>::value_type
SlidingWindowManager<MODEL>::compute_trace_braket(int braket,
                                                  std::pair<op_it_t, op_it_t> ops_range, double tau_left,
                                                  double tau_right) const {
  BRAKET_TYPE ket = right_states[braket].back();
  evolve_ket(*p_model, ket, ops_range, tau_right, tau_left);
  if (left_states[braket].back().sector() == ket.sector()) {
    return p_model->product(left_states[braket].back(), ket);
  } else {
    return 0.0;
  }
}

template<typename T>
struct bound_greater: std::binary_function<std::pair<T, int>, std::pair<T, int>, bool> {
  bool operator()(const std::pair<T, int> &x, const std::pair<T, int> &y) const {
    return x.first > y.first;
  }
};

template<typename MODEL>
std::pair<bool, typename ExtendedScalar<typename model_traits<MODEL>::SCALAR_T>::value_type>
SlidingWindowManager<MODEL>::lazy_eval_trace(EXTENDED_REAL trace_cutoff,
                                             std::vector<EXTENDED_REAL> &trace_bound) const {
  namespace bll = boost::lambda;

  sanity_check();

  OperatorTime tau_right = get_op_tau_low(position_right_edge);
  OperatorTime tau_left = get_op_tau_high(position_left_edge);
  std::pair<op_it_t, op_it_t> ops_range = operators.range(tau_right <= bll::_1, bll::_1 <= tau_left);

  EXTENDED_REAL trace_bound_current = std::accumulate(trace_bound.begin(), trace_bound.end(), EXTENDED_REAL(0.0));
  assert(trace_bound_current >= 0.0);

  //sort trace bounds in decreasing order
  std::vector<std::pair<EXTENDED_REAL, int> > indices(num_brakets);
  for (int braket = 0; braket < num_brakets; ++braket) {
    indices[braket] = std::make_pair(trace_bound[braket], braket);
  }
  std::sort(indices.begin(), indices.end(), bound_greater<EXTENDED_REAL>());
  for (int idx = 0; idx < num_brakets - 1; ++idx) {
    check_true(indices[idx].first >= indices[idx].first);
  }

  EXTENDED_SCALAR trace_sum = 0.0;
  for (int idx = 0; idx < num_brakets; ++idx) {
    int braket = indices[idx].second;
    if (trace_bound[braket] < 1E-15 * myabs(trace_sum)) {
      break;
    }

    const EXTENDED_SCALAR trace_braket = compute_trace_braket(braket, ops_range, tau_left.time(), tau_right.time());

    assert(myabs(trace_braket) <= trace_bound[braket] * 1.01);
    trace_sum += trace_braket;
    trace_bound[braket] = myabs(trace_braket);
    trace_bound_current = std::accumulate(trace_bound.begin(), trace_bound.end(), EXTENDED_REAL(0.0));
    if (trace_bound_current < trace_cutoff) {
      return std::make_pair(false, 0.0);
    }
  }
  return std::make_pair(myabs(trace_sum) > trace_cutoff, trace_sum);
}

template<typename MODEL>
void
SlidingWindowManager<MODEL>::evolve_bra(const MODEL &model, BRAKET_TYPE &bra,
            std::pair<op_it_t, op_it_t> ops_range, double tau_old, double tau_new) {
  if (bra.invalid()) {
    return;
  }

  bra.normalize();

  //range check
  check_true(tau_new <= tau_old);

  const int num_ops = std::distance(ops_range.first, ops_range.second);

  if (num_ops > 0) {
    op_it_t it = ops_range.second;
    --it;//pointing to the operator with the largest tau
    op_it_t it_down = it;
    --it_down;

    model.sector_propagate_bra(bra, tau_old - it->time());
    for (int i = 0; i < num_ops; i++) {
      model.apply_op_hyb_bra(it->type(), it->flavor(), bra);

      if (it == ops_range.first) {//no more operators left
        model.sector_propagate_bra(bra, it->time() - tau_new);
      } else {
        //there is still some operator left
        model.sector_propagate_bra(bra, it->time() - it_down->time());
      }
      --it;
      --it_down;
    }
  } else {
    model.sector_propagate_bra(bra, tau_old - tau_new);
  }
  bra.normalize();
}

template<typename MODEL>
void
SlidingWindowManager<MODEL>::evolve_ket(const MODEL &model, BRAKET_TYPE &ket,
                                        std::pair<op_it_t, op_it_t> ops_range, double tau_old, double tau_new) {
  if (ket.invalid()) {
    return;
  }

  //range check
  check_true(tau_new >= tau_old);

  ket.normalize();

  auto num_ops = std::distance(ops_range.first, ops_range.second);
  if (num_ops > 0) {
    op_it_t it = ops_range.first;
    op_it_t it_up = it;
    it_up++;

    BRAKET_TYPE tmp;
    model.sector_propagate_ket(ket, it->time() - tau_old);
    for (auto i = 0; i < num_ops; i++) {
      model.apply_op_hyb_ket(it->type(), it->flavor(), ket);

      if (it_up != ops_range.second) {
        model.sector_propagate_ket(ket, it_up->time() - it->time());
      } else {
        model.sector_propagate_ket(ket, tau_new - it->time());
      }
      it++;
      it_up++;
    }
  } else {
    model.sector_propagate_ket(ket, tau_new - tau_old);
  }

  ket.normalize();
}

template<typename MODEL>
EXTENDED_REAL
SlidingWindowManager<MODEL>::compute_trace_bound(std::vector<EXTENDED_REAL> &bound) const {
  namespace bll = boost::lambda;
  OperatorTime tau_right = get_op_tau_low(position_right_edge);
  OperatorTime tau_left = get_op_tau_high(position_left_edge);
  std::pair<op_it_t, op_it_t> ops_range = operators.range(tau_right <= bll::_1, bll::_1 <= tau_left);
  std::vector<psi> ops_in_range(ops_range.first, ops_range.second);
  const int num_ops = ops_in_range.size();

  assert(tau_left >= tau_right);
  assert(bound.size() >= get_num_brakets());
  std::fill(bound.begin(), bound.end(), 0.0);
  EXTENDED_REAL trace_bound_sum = 0.0;
  for (int braket = 0; braket < num_brakets; ++braket) {
    if (is_braket_invalid(braket)) {
      continue;
    }

    int min_dim = right_states[braket].back().min_dim();
    int sector_ket = right_states[braket].back().sector();
    EXTENDED_REAL norm_prod = 1.0;

    if (num_ops > 0) {
      std::vector<psi>::const_iterator it = ops_in_range.begin();
      std::vector<psi>::const_iterator it_up = it;
      it_up++;

      assert(sector_ket >= 0);
      norm_prod *= compute_exp(sector_ket, it->time() - tau_right);
      for (int i = 0; i < num_ops; i++) {
        assert(sector_ket >= 0);
        sector_ket = p_model->get_dst_sector_ket(it->type(), it->flavor(), sector_ket);
        if (sector_ket == nirvana) {
          break;
        }
        min_dim = std::min(min_dim, p_model->dim_sector(sector_ket));

        if (it_up != ops_in_range.end()) {
          norm_prod *= compute_exp(sector_ket, it_up->time() - it->time());
        } else {
          norm_prod *= compute_exp(sector_ket, tau_left - it->time());
        }
        it++;
        it_up++;
      }
      if (sector_ket == nirvana) {
        norm_prod = 0.0;
      }
    } else {
      norm_prod *= compute_exp(sector_ket, tau_left - tau_right);
    }
    bound[braket] =
        sector_ket == left_states[braket].back().sector() ?
        norm_prod * norm_left_states[braket].back() * norm_right_states[braket].back() *
            ((EXTENDED_REAL) 1. * min_dim) :
        0.0;
    trace_bound_sum += bound[braket];
  }
  check_true(trace_bound_sum >= 0.0);
  return trace_bound_sum;
}


template<typename MODEL>
void
SlidingWindowManager<MODEL>::move_window_to_next_position() {
  sanity_check();

  // If the window width is beta, no way to move it.
  if (position_left_edge == n_section && position_right_edge == 0) {
    return;
  }

  // First check if the direction is OK.
  // Reverse the move direction otherwise.
  if (direction_move_local_window == ITIME_LEFT && position_left_edge == n_section) {
    direction_move_local_window = ITIME_RIGHT;
  } else if (direction_move_local_window == ITIME_RIGHT && position_right_edge == 0) {
    direction_move_local_window = ITIME_LEFT;
  }

  move_window_to(direction_move_local_window);

  sanity_check();
}

template<typename MODEL>
void
SlidingWindowManager<MODEL>::move_window_to(ITIME_AXIS_LEFT_OR_RIGHT which_direction) {
  sanity_check();
  if (which_direction == ITIME_LEFT) {
    move_backward_edge(ITIME_LEFT);
    move_forward_right_edge();
  } else if (which_direction == ITIME_RIGHT) {
    move_backward_edge(ITIME_RIGHT);
    move_forward_left_edge();
  } else {
    throw std::runtime_error("unknown direction");
  }
  sanity_check();
}

template<typename MODEL>
void SlidingWindowManager<MODEL>::pop_back_bra(int num_pop_back) {
  const int new_size = depth_left_states() - num_pop_back;
  for (int braket = 0; braket < num_brakets; ++braket) {
    left_states[braket].resize(new_size);
    norm_left_states[braket].resize(new_size);
  }
}

template<typename MODEL>
void SlidingWindowManager<MODEL>::pop_back_ket(int num_pop_back) {
  const int new_size = depth_right_states() - num_pop_back;
  for (int braket = 0; braket < num_brakets; ++braket) {
    right_states[braket].resize(new_size);
    norm_right_states[braket].resize(new_size);
  }
}

template<typename MODEL>
void SlidingWindowManager<MODEL>::restore_state(state_t state) {
  set_uniform_mesh(
      boost::get<3>(state),
      boost::get<1>(state),
      boost::get<2>(state),
      boost::get<0>(state)
    );
  move_left_edge_to(boost::get<0>(state));
  check_true(get_position_right_edge() == boost::get<1>(state));
  check_true(get_position_left_edge() == boost::get<0>(state));
};

template<typename MODEL>
void
SlidingWindowManager<MODEL>::sanity_check() const {
  check_true(operators.size()%2 == 0, "Number of operators in sw must be even!");
  check_true(get_position_left_edge() >= get_position_right_edge());
  for (int braket = 0; braket < num_brakets; ++braket) {
    check_true(left_states[braket].size() == depth_left_states());
    check_true(right_states[braket].size() == depth_right_states());
    check_true(norm_left_states[braket].size() == depth_left_states());
    check_true(norm_right_states[braket].size() == depth_right_states());
    check_true(norm_right_states[braket].back() >= 0);
    check_true(norm_left_states[braket].back() >= 0);
  }
}
