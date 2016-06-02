#include "../sliding_window.hpp"

template<typename MODEL>
SlidingWindowManager<MODEL>::SlidingWindowManager(MODEL* p_model_, double beta)
        : p_model(p_model_),
          BETA(beta),
          num_brakets(p_model->num_brakets()),
          norm_cutoff(std::sqrt(std::numeric_limits<double>::min())) {};

template<typename MODEL>
void
SlidingWindowManager<MODEL>::init_stacks(int n_window_size, const operator_container_t& operators)
{
    left_states.resize(num_brakets);
    right_states.resize(num_brakets);
    norm_left_states.resize(num_brakets);//for bra
    norm_right_states.resize(num_brakets);//for ket
    for (int braket=0; braket<num_brakets; ++braket) {
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

    set_window_size(n_window_size, operators);
    sanity_check();
}

template<typename MODEL>
void SlidingWindowManager<MODEL>::set_window_size(int n_window_new, const operator_container_t& operators,
                                                  int new_position_right_edge, ITIME_AXIS_LEFT_OR_RIGHT new_direction_move)
{
    assert(n_window_new>0);
    sanity_check();

    //reset
    while (depth_right_states()>1) {
        move_backward_edge(ITIME_RIGHT);
    }
    while (depth_left_states()>1) {
        move_backward_edge(ITIME_LEFT);
    }

    n_window = n_window_new;
    if (n_window >= 2) {
        position_right_edge = 0;
        position_left_edge = 2*n_window;

        for (int i=0; i<new_position_right_edge; ++i) {
            move_forward_right_edge(operators);
        }
        for (int i=0; i<2*n_window-2-new_position_right_edge; ++i) {
            move_forward_left_edge(operators);
        }
        assert(position_left_edge-position_right_edge==2);
        assert(position_right_edge==new_position_right_edge);

        direction_move_local_window = new_direction_move;
        if (get_position_right_edge()==0) {
            direction_move_local_window = ITIME_LEFT;
        } else if (get_position_right_edge()==2*get_n_window()-2) {
            direction_move_local_window = ITIME_RIGHT;
        }
    } else {
        position_right_edge = 0;
        position_left_edge = 2*n_window;
    }
    sanity_check();
}

template<typename MODEL>
void
SlidingWindowManager<MODEL>::move_backward_edge(ITIME_AXIS_LEFT_OR_RIGHT which_edge, int num_move)
{
    sanity_check();
    if (which_edge==ITIME_RIGHT) {
        if (static_cast<int>(position_right_edge)-num_move<0) {
            throw std::runtime_error("Out of range in move_backward_edge");
        }
        pop_back_ket(num_move);
        position_right_edge -= num_move;
    } else if(which_edge==ITIME_LEFT) {
        if (position_left_edge+num_move>2*n_window) {
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
SlidingWindowManager<MODEL>::move_forward_right_edge(const operator_container_t& operators, int num_move)
{
    namespace bll = boost::lambda;
    sanity_check();

    for (int move=0; move<num_move; ++move) {
        //range check
        assert(position_right_edge>=0);
        assert(position_right_edge<=2*n_window-2);
        const double tau_edge_old = get_tau_edge(position_right_edge);
        const double tau_edge_new = get_tau_edge(position_right_edge+1);
        //const int new_size = depth_right_states()+1;
        std::pair<op_it_t,op_it_t> ops_range = operators.range(tau_edge_old<=bll::_1, bll::_1<tau_edge_new);
        EXTENDED_REAL max_norm = -1;
        for (int i_braket = 0; i_braket < num_brakets; ++i_braket) {
            right_states[i_braket].push_back(right_states[i_braket].back());
            evolve_ket(*p_model, right_states[i_braket].back(), ops_range, tau_edge_old, tau_edge_new);
            norm_right_states[i_braket].push_back(right_states[i_braket].back().compute_spectral_norm());
            if (max_norm < norm_right_states[i_braket].back()) {
                max_norm = norm_right_states[i_braket].back();
            }
        }
        for (int i_braket = 0; i_braket < num_brakets; ++i_braket) {
            if (norm_right_states[i_braket].back()<max_norm*1E-100) {
                norm_right_states[i_braket].back() = 0.0;
                right_states[i_braket].back().set_invalid();
            }
        }
        ++position_right_edge;
        sanity_check();
    }
}


//large-beta side
template<typename MODEL>
void
SlidingWindowManager<MODEL>::move_forward_left_edge(const operator_container_t& operators_tmp, int num_move)
{
    namespace bll = boost::lambda;

    for (int move=0; move<num_move; ++move) {
        //range check
        assert(position_left_edge >= 2);
        assert(position_left_edge <= 2 * n_window);

        sanity_check();

        const double tau_edge_old = get_tau_edge(position_left_edge);
        const double tau_edge_new = get_tau_edge(position_left_edge - 1);
        const std::pair<op_it_t, op_it_t> ops_range = operators_tmp.range(tau_edge_new < bll::_1,
                                                                          bll::_1 <= tau_edge_old);
        //const int num_ops = std::distance(ops_range.first, ops_range.second);

        EXTENDED_REAL max_norm = -1;
        for (int i_braket = 0; i_braket < num_brakets; ++i_braket) {
            left_states[i_braket].push_back(left_states[i_braket].back());
            evolve_bra(*p_model, left_states[i_braket].back(), ops_range, tau_edge_old, tau_edge_new);
            norm_left_states[i_braket].push_back(left_states[i_braket].back().compute_spectral_norm());
            if (max_norm < norm_left_states[i_braket].back()) {
                max_norm = norm_left_states[i_braket].back();
            }
        }
        for (int i_braket = 0; i_braket < num_brakets; ++i_braket) {
            if (norm_left_states[i_braket].back()<max_norm*1E-100) {
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
SlidingWindowManager<MODEL>::move_right_edge_to(const operator_container_t& operators, int pos)
{
    assert(pos>=0 && pos<=2*n_window);
    if (get_position_right_edge()>pos) {
        move_backward_edge(ITIME_RIGHT, get_position_right_edge()-pos);
    } else if (get_position_right_edge()<pos) {
        move_forward_right_edge(operators, pos-get_position_right_edge());
    }
}

template<typename MODEL>
void
SlidingWindowManager<MODEL>::move_left_edge_to(const operator_container_t& operators, int pos)
{
    assert(pos>=0 && pos<=2*n_window);
    const int current_pos = get_position_left_edge();
    if (current_pos>pos) {
        move_forward_left_edge(operators, current_pos-pos);
    } else if (current_pos<pos) {
        move_backward_edge(ITIME_LEFT, pos-current_pos);
    }
}


template<typename MODEL>
typename ExtendedScalar<typename model_traits<MODEL>::SCALAR_T>::value_type
SlidingWindowManager<MODEL>::compute_trace(const operator_container_t& operators) const {
    namespace bll = boost::lambda;

    sanity_check();

    EXTENDED_SCALAR trace = 0.0;

    const double tau_right = get_tau_edge(position_right_edge);
    const double tau_left = get_tau_edge(position_left_edge);
    std::pair<op_it_t,op_it_t> ops_range = operators.range(tau_right<=bll::_1, bll::_1<=tau_left);

    for (int i_braket = 0; i_braket < num_brakets; ++i_braket) {
        if (is_braket_invalid(i_braket)) {
          continue;
        }
        BRAKET_TYPE ket = right_states[i_braket].back();

        evolve_ket(*p_model, ket, ops_range, tau_right, tau_left);
        if (left_states[i_braket].back().sector()==ket.sector()) {
            const EXTENDED_SCALAR trace_braket = p_model->product(left_states[i_braket].back(), ket);
            /*
#ifndef NDEBUG
            const typename BRAKET_TYPE::EXTENDED_REAL trace_bound = std::min(left_states[i_braket].back().min_dim(), ket.min_dim())
               *left_states[i_braket].back().compute_spectral_norm()
               *ket.compute_spectral_norm();
            if (!(trace_bound>=0)) {
               std::cout << "trace_bound " << trace_bound << std::endl;
               std::cout << "min_dim " << std::min(left_states[i_braket].back().min_dim(), ket.min_dim()) << std::endl;
               std::cout << "left_norm " << left_states[i_braket].back().compute_spectral_norm() << std::endl;
               std::cout << "right_norm " << ket.compute_spectral_norm() << std::endl;
            }
            assert(trace_bound>=0);
            const double eps = std::max(1E-8*std::max(std::abs(trace_braket),trace_bound),1E-8);
            assert(std::abs(trace_braket)<=trace_bound+eps);
#endif
             */
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
    if (left_states[braket].back().sector()==ket.sector()) {
        return p_model->product(left_states[braket].back(), ket);
    } else {
        return 0.0;
    }
}

template<typename T>
struct bound_greater : std::binary_function <std::pair<T,int>,std::pair<T,int>,bool> {
  bool operator() (const std::pair<T,int>& x, const std::pair<T,int>& y) const {
      return x.first>y.first;
  }
};

template<typename MODEL>
std::pair<bool, typename ExtendedScalar<typename model_traits<MODEL>::SCALAR_T>::value_type>
SlidingWindowManager<MODEL >::lazy_eval_trace(const operator_container_t& operators, EXTENDED_REAL trace_cutoff,
                                                            std::vector<EXTENDED_REAL>& trace_bound) const {
    namespace bll = boost::lambda;

    sanity_check();

    const double tau_right = get_tau_edge(position_right_edge);
    const double tau_left = get_tau_edge(position_left_edge);
    std::pair<op_it_t,op_it_t> ops_range = operators.range(tau_right<=bll::_1, bll::_1<=tau_left);

    EXTENDED_REAL trace_bound_current = std::accumulate(trace_bound.begin(), trace_bound.end(), EXTENDED_REAL(0.0) );
    assert(trace_bound_current>0.0);

    //sort trace bounds in decreasing order
    std::vector<std::pair<EXTENDED_REAL, int> > indices(num_brakets);
    for (int braket=0; braket<num_brakets; ++braket){
        indices[braket] = std::make_pair(trace_bound[braket],braket);
    }
    std::sort(indices.begin(),indices.end(),bound_greater<EXTENDED_REAL>());
#ifndef NDEBUG
    for (int idx=0; idx<num_brakets-1; ++idx){
        assert(indices[idx].first>=indices[idx].first);
    }
#endif

    EXTENDED_SCALAR trace_sum = 0.0;
    for (int idx=0; idx<num_brakets; ++idx) {
        int braket = indices[idx].second;
        if (trace_bound[braket]<1E-15*myabs(trace_sum)) {
            break;
        }

        const EXTENDED_SCALAR trace_braket = compute_trace_braket(braket, ops_range, tau_left, tau_right);

        assert(myabs(trace_braket)<=trace_bound[braket]*1.01);
        trace_sum += trace_braket;
        trace_bound[braket] = myabs(trace_braket);
        trace_bound_current = std::accumulate(trace_bound.begin(), trace_bound.end(), EXTENDED_REAL(0.0) );
        if (trace_bound_current<trace_cutoff) {
            return std::make_pair(false,0.0);
        }
    }
    return std::make_pair(myabs(trace_sum)>trace_cutoff, trace_sum);
}

template<typename MODEL>
void
SlidingWindowManager<MODEL>::evolve_bra(const MODEL& model, BRAKET_TYPE& bra,
                                                       std::pair<op_it_t,op_it_t> ops_range, double tau_old, double tau_new) {
    if (bra.invalid()) {
        return;
    }

    //range check
    assert(tau_new<=tau_old);

    const int num_ops = std::distance(ops_range.first,ops_range.second);

    if (num_ops > 0) {
        op_it_t it = ops_range.second;
        --it;//pointing to the operator with the largest tau
        op_it_t it_down = it;
        --it_down;

        model.sector_propagate_bra(bra, tau_old-it->time());
        for (int i = 0; i < num_ops; i++) {
            model.apply_op_hyb_bra(it->type(), it->flavor(), bra);

            if (it == ops_range.first) {//no more operators left
                model.sector_propagate_bra(bra, it->time()-tau_new);
            } else {
                //there is still some operator left
                model.sector_propagate_bra(bra, it->time()-it_down->time());
            }
            --it;
            --it_down;
        }
    } else {
        model.sector_propagate_bra(bra, tau_old-tau_new);
    }
}

template<typename MODEL>
void
SlidingWindowManager<MODEL>::evolve_ket(const MODEL& model, BRAKET_TYPE& ket,
                                                        std::pair<op_it_t,op_it_t> ops_range, double tau_old, double tau_new) {
    if (ket.invalid()) {
        return;
    }

    //range check
    assert(tau_new>=tau_old);

    const int num_ops = std::distance(ops_range.first,ops_range.second);
    if (num_ops > 0) {
        op_it_t it = ops_range.first;
        op_it_t it_up = it;
        it_up++;

        BRAKET_TYPE tmp;
        model.sector_propagate_ket(ket, it->time()-tau_old);
        for (int i = 0; i < num_ops; i++) {
            model.apply_op_hyb_ket(it->type(), it->flavor(), ket);

            if (it_up != ops_range.second) {
                model.sector_propagate_ket(ket, it_up->time()-it->time());
            } else {
                model.sector_propagate_ket(ket, tau_new-it->time());
            }
            it++;
            it_up++;
        }
    } else {
        model.sector_propagate_ket(ket, tau_new-tau_old);
    }
}

template<typename MODEL>
EXTENDED_REAL
SlidingWindowManager<MODEL>::compute_trace_bound(const operator_container_t& operators,
                                                                std::vector<EXTENDED_REAL>& bound) const {
    namespace bll = boost::lambda;
    const double tau_right = get_tau_edge(position_right_edge);
    const double tau_left = get_tau_edge(position_left_edge);
    std::pair<op_it_t,op_it_t> ops_range = operators.range(tau_right<=bll::_1, bll::_1<=tau_left);

    assert(tau_left>=tau_right);
    assert(bound.size()>=get_num_brakets());
    std::fill(bound.begin(), bound.end(), 0.0);
    EXTENDED_REAL trace_bound_sum = 0.0;
    for (int braket = 0; braket<num_brakets; ++braket) {
        if (is_braket_invalid(braket)) {
            continue;
        }

        int min_dim = right_states[braket].back().min_dim();
        int sector_ket = right_states[braket].back().sector();
        EXTENDED_REAL norm_prod = 1.0;

        const int num_ops = std::distance(ops_range.first,ops_range.second);
        if (num_ops > 0) {
            op_it_t it = ops_range.first;
            op_it_t it_up = it;
            it_up++;

            assert(sector_ket>=0);
            norm_prod *= compute_exp(sector_ket, it->time()-tau_right);
            for (int i = 0; i < num_ops; i++) {
                assert(sector_ket>=0);
                sector_ket = p_model->get_dst_sector_ket(it->type(), it->flavor(), sector_ket);
                if (sector_ket==nirvana) {
                    break;
                }
                min_dim = std::min(min_dim, p_model->dim_sector(sector_ket));

                if (it_up != ops_range.second) {
                    norm_prod *= compute_exp(sector_ket, it_up->time()-it->time());
                } else {
                    norm_prod *= compute_exp(sector_ket, tau_left-it->time());
                }
                it++;
                it_up++;
            }
            if (sector_ket==nirvana) {
                norm_prod = 0.0;
            }
        } else {
            norm_prod *= compute_exp(sector_ket, tau_left-tau_right);
        }
        bound[braket] = norm_prod*norm_left_states[braket].back()*norm_right_states[braket].back()*min_dim;
        trace_bound_sum += bound[braket];
    }
    return trace_bound_sum;
}


template<typename MODEL>
void
SlidingWindowManager<MODEL>::move_window_to_next_position(const operator_container_t& operators)
{
    sanity_check();

    if (n_window == 1) {
        return;
    }

    if (direction_move_local_window == ITIME_LEFT) {
        if (position_left_edge == 2*n_window) {
            direction_move_local_window = ITIME_RIGHT;
            move_window_to(operators, ITIME_RIGHT);
        } else {
            move_window_to(operators, ITIME_LEFT);
        }
    } else {
        if (position_right_edge == 0) {
            direction_move_local_window = ITIME_LEFT;
            move_window_to(operators, ITIME_LEFT);
        } else {
            move_window_to(operators, ITIME_RIGHT);
        }
    }

    sanity_check();
}

template<typename MODEL>
void
SlidingWindowManager<MODEL>::move_window_to(const operator_container_t& operators, ITIME_AXIS_LEFT_OR_RIGHT which_direction)
{
    sanity_check();
    if (which_direction==ITIME_LEFT) {
        move_forward_right_edge(operators);
        move_backward_edge(ITIME_LEFT);
    } else if (which_direction==ITIME_RIGHT) {
        move_backward_edge(ITIME_RIGHT);
        move_forward_left_edge(operators);
    } else {
        throw std::runtime_error("unknown direction");
    }
    sanity_check();
}

template<typename MODEL>
void SlidingWindowManager<MODEL>::pop_back_bra(int num_pop_back) {
    const int new_size = depth_left_states()-num_pop_back;
    for (int braket=0; braket<num_brakets; ++braket) {
        left_states[braket].resize(new_size);
        norm_left_states[braket].resize(new_size);
    }
}

template<typename MODEL>
void SlidingWindowManager<MODEL>::pop_back_ket(int num_pop_back) {
    const int new_size = depth_right_states()-num_pop_back;
    for (int braket=0; braket<num_brakets; ++braket) {
        right_states[braket].resize(new_size);
        norm_right_states[braket].resize(new_size);
    }
}

template<typename MODEL>
void SlidingWindowManager<MODEL>::restore_state(const operator_container_t& ops, state_t state) {
    set_window_size(boost::get<3>(state), ops, boost::get<1>(state), boost::get<2>(state));
    move_left_edge_to(ops, boost::get<0>(state));
    assert(get_position_right_edge()==boost::get<1>(state));
    assert(get_position_left_edge()==boost::get<0>(state));
    //move_right_edge_to(ops, boost::get<1>(state));
    //direction_move_local_window = boost::get<2>(state);
};

template<typename MODEL>
void
SlidingWindowManager<MODEL>::sanity_check() const {
#ifndef NDEBUG
    for (int braket=0; braket<num_brakets; ++braket) {
        assert(left_states[braket].size()==depth_left_states());
        assert(right_states[braket].size()==depth_right_states());
        assert(norm_left_states[braket].size()==depth_left_states());
        assert(norm_right_states[braket].size()==depth_right_states());
    }
#endif
}
