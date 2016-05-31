#pragma once

#include <cmath>
#include <limits.h>
#include <math.h>

#include <boost/tuple/tuple.hpp>
#include <boost/assert.hpp>

#include "update.hpp"
#include "operator_util.hpp"

template<class SCALAR>
bool equal_det(SCALAR det1, SCALAR det2) {
    const double eps = 1E-8;
    if (std::abs(det1)<eps && std::abs(det2)<eps) {
        return true;
    }
    return (std::abs((det2-det1)/det1)<eps);
}

/**
 * Compute sign of the permutation which time-orders \psidag_a1(\tau_1)\psi_a1(\tau'_1)...\psidag_ak(\tau_nk)\psi_ak(\tau'_nk)
 * This is done by counting the number of operators that are in (t_ins, t_rem)
 */
inline int compute_permutation_change(const operator_container_t& operators, double t_ins, double t_rem)
{
    namespace bll = boost::lambda;
    typedef operator_container_t::iterator it_t;

    int perm_number = 0;

    double max, min;
    if (t_ins > t_rem) {
        perm_number += 1;
        max = t_ins;
        min = t_rem;
    } else {
        max = t_rem;
        min = t_ins;
    }
    std::pair<it_t,it_t> p = operators.range(min<bll::_1, bll::_1<max);
    perm_number += std::distance(p.first, p.second);

    return (perm_number%2==0 ? 1 : -1);
}

inline void operators_insert_nocopy(operator_container_t &operators, double t_ins, double t_rem, int flavor_ins, int flavor_rem)
{
    const psi op_ins(t_ins, CREATION_OP, flavor_ins);
    const psi op_rem(t_rem, ANNIHILATION_OP, flavor_rem);

    safe_insert(operators,op_ins);
    safe_insert(operators,op_rem);
}

inline std::pair<psi, psi> operators_remove_nocopy(operator_container_t &operators, double t_ins, double t_rem, int flavor_ins, int flavor_rem)
{

    const psi op_ins = psi(t_ins,CREATION_OP,flavor_ins);
    const psi op_rem = psi(t_rem,ANNIHILATION_OP,flavor_rem);

    safe_erase(operators, op_ins);
    safe_erase(operators, op_rem);

    return std::make_pair(op_ins, op_rem); //return both operators
}


// update_type, accepted, distance, acceptance probability, valid_move_generated
// Note: this function is too long. Better to split into several functions.
template<typename SCALAR, typename R, typename M_TYPE, typename SLIDING_WINDOW>
boost::tuple<int,bool,double,SCALAR,bool> insert_remove_pair_flavor(R& rng, int creation_flavor, int annihilation_flavor, SCALAR & det, double BETA,
        std::vector<int> & order_creation_flavor, std::vector<int> & order_annihilation_flavor, operator_container_t& creation_operators,
        operator_container_t& annihilation_operators, M_TYPE& M, SCALAR & sign, SCALAR & trace, operator_container_t& operators,
        double cutoff,
        SLIDING_WINDOW& sliding_window,
        unsigned int max_order=UINT_MAX
        )
{
    namespace bll = boost::lambda;

    // insert and remove random pair (times and flavors)
    static std::vector<double> trace_bound(sliding_window.get_num_brakets());
    bool valid_move_generated = false;

    int flavor_ins = creation_flavor;
    int flavor_rem = annihilation_flavor;
    const double tau_low = sliding_window.get_tau_low();
    const double tau_high = sliding_window.get_tau_high();

    const double r_th = rng();

    if (rng()<0.5) { // try to insert a pair

        if (operators.size()>=max_order) {
            return boost::make_tuple(0,false,0.0,0.0,false);
        }

        const double t_ins=open_random(rng,tau_low,tau_high);
        const double t_rem_max = std::min(t_ins+cutoff,tau_high);
        const double t_rem_min = std::max(t_ins-cutoff,tau_low);
        const double t_rem=open_random(rng,t_rem_max,t_rem_min);

        const psi op_ins(t_ins, CREATION_OP, flavor_ins);
        const psi op_rem(t_rem, ANNIHILATION_OP, flavor_rem);

        //check_consistency_operators(operators, creation_operators, annihilation_operators);
        if (operators.find(op_ins)!=operators.end() || operators.find(op_rem)!=operators.end() || t_ins==t_rem) {
           return boost::make_tuple(0,false,0.0,0.0,false);
        }

        assert(std::abs(t_ins-t_rem)<=cutoff);

        valid_move_generated = true;

        bool err;
        const int num_pairs_after_insertion = count_num_pairs_after_insert(operators, creation_operators, annihilation_operators,
                                              flavor_ins, flavor_rem,
                                              tau_high, tau_low, cutoff, t_ins, t_rem, err);//should be revised
        if (err) {
            return boost::make_tuple(0,false,0.0,0.0,false);
        }

        safe_insert(operators,op_ins);
        safe_insert(operators,op_rem);

        const double trace_bound_sum = sliding_window.compute_trace_bound(operators, trace_bound);
        if (trace_bound_sum==0.0) {
            safe_erase(operators,op_ins);
            safe_erase(operators,op_rem);
            return boost::make_tuple(0,false,std::abs(t_ins-t_rem),0.0,valid_move_generated);
        }

        int column=0;   // creation operator position
        for (operator_container_t::iterator it=creation_operators.begin(); it!=creation_operators.end(); it++) {
            if (it->time()<t_ins) {
                column++;
            } else {
                break;
            }
        }
        int row=0;		// annihilation operator position
        for (operator_container_t::iterator it=annihilation_operators.begin(); it!=annihilation_operators.end(); it++) {
            if (it->time()<t_rem) {
                row++;
            } else {
                break;
            }
        }

        double flavor_sign=((row+column)%2==0 ? 1 : -1);

        Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic>
          sign_Fs(creation_operators.size(),1), Fe_M(1,annihilation_operators.size());
        SCALAR det_rat = det_rat_row_column_up(flavor_ins, flavor_rem, t_ins, t_rem,
                                               row, column, M, creation_operators, annihilation_operators,
                                               sliding_window.get_p_model()->get_F(), sign_Fs, Fe_M, BETA);
        const int perm_change = compute_permutation_change(operators, t_ins, t_rem);

        bool accepted;
        SCALAR trace_new, prob;
        if (det_rat!=0.0) {
            const SCALAR rest = (((tau_high-tau_low)*(t_rem_max-t_rem_min))/num_pairs_after_insertion)*det_rat*(1.*perm_change)*(1.*flavor_sign);
            const double trace_cutoff = std::abs(r_th*trace/rest);
            boost::tie(accepted,trace_new) = sliding_window.lazy_eval_trace(operators, trace_cutoff, trace_bound);
            prob = rest*(trace_new/trace);
            assert(accepted==std::abs(prob)>r_th);
        } else {
            accepted = false;
            trace_new = 0.0;
            prob = 0.0;
        }

        if (accepted) { // move accepted
            //std::cout << " accepted " << std::endl;
            safe_insert(creation_operators,op_ins);
            safe_insert(annihilation_operators,op_rem);

            order_creation_flavor[flavor_ins] += 1;
            order_annihilation_flavor[flavor_rem] += 1;
            compute_M_row_column_up(row, column, M, sign_Fs, Fe_M, det_rat);

            det *= det_rat;
            sign *= prob/std::abs(prob);
            trace=trace_new;
            return boost::make_tuple(0,true,std::abs(t_ins-t_rem),prob,valid_move_generated);
        } else { // rejected
            safe_erase(operators, op_ins);
            safe_erase(operators, op_rem);
            return boost::make_tuple(0,false,std::abs(t_ins-t_rem),prob,valid_move_generated);
        }
    }

    else if(order_creation_flavor[creation_flavor]>0 && order_annihilation_flavor[annihilation_flavor]>0) { // try to remove a pair
        boost::tuple<int,operator_container_t::iterator,operator_container_t::iterator> r =
            pick_up_pair(rng, creation_operators, annihilation_operators, flavor_ins, flavor_rem, tau_high, tau_low, cutoff, BETA);
        const int num_pairs_old = r.get<0>();
        if (num_pairs_old==0) {
            return boost::make_tuple(1,false,0.0,0.0,valid_move_generated);
        }
        const operator_container_t::iterator it_c = r.get<1>();
        const operator_container_t::iterator it_a = r.get<2>();
        const int position_c = std::distance(creation_operators.begin(), it_c);
        const int position_a = std::distance(annihilation_operators.begin(), it_a);
        const psi op_c(*it_c);
        const psi op_a(*it_a);
        assert(std::abs(it_c->time()-it_a->time())<=cutoff);
        assert(it_c->time()<=tau_high&&it_c->time()>=tau_low);
        assert(it_a->time()<=tau_high&&it_a->time()>=tau_low);

        valid_move_generated = true;

        const double c_time=it_c->time();
        const double a_time=it_a->time();
        const double time_diff = std::abs(c_time-a_time);

        std::pair<psi, psi> removed_ops= operators_remove_nocopy(operators, it_c->time(), it_a->time(), it_c->flavor(), it_a->flavor());

        // Caution have to compute perm_change after removing operators
        int perm_change=compute_permutation_change(operators, c_time, a_time);

        const double trace_bound_sum = sliding_window.compute_trace_bound(operators, trace_bound);
        if (trace_bound_sum==0.0) {
            safe_insert(operators,removed_ops.second); //remove annihilator
            safe_insert(operators,removed_ops.first); //remove creator
            return boost::make_tuple(1,false,time_diff,0.0,valid_move_generated);
        }

        // it is possible to remove the pair
        double flavor_sign = ((position_c+position_a)%2==0 ? 1 : -1);

        const SCALAR det_rat = det_rat_row_column_down(position_c, position_a, M);

        const double t_ins_range_in_reverse_process = tau_high-tau_low;
        const double t_rem_range_in_reverse_process = std::min(c_time+cutoff,tau_high)-std::max(c_time-cutoff,tau_low);

        bool accepted;
        SCALAR trace_new, prob;
        if (det_rat!=0.0) {
            const SCALAR rest = ((1.*num_pairs_old)/(t_ins_range_in_reverse_process*t_rem_range_in_reverse_process))*
                                det_rat*(1.*perm_change)*(1.*flavor_sign);
            const double trace_cutoff = std::abs(r_th*trace/rest);
            boost::tie(accepted,trace_new) = sliding_window.lazy_eval_trace(operators, trace_cutoff, trace_bound);
            prob = rest*(trace_new/trace);
            assert(accepted==std::abs(prob)>r_th);
        } else {
            accepted = false;
            trace_new = 0.0;
            prob = 0.0;
        }

        if (r_th < std::abs(prob)) { // move accepted
            det *= det_rat;

            sign *= prob/std::abs(prob);

            order_creation_flavor[it_c->flavor()] -= 1;
            order_annihilation_flavor[it_a->flavor()] -= 1;

            compute_M_row_column_down(position_c, position_a, M);

            safe_erase(creation_operators,op_c);
            safe_erase(annihilation_operators,op_a);

            trace=trace_new;
            return boost::make_tuple(1,true,time_diff,prob,valid_move_generated);
        } else { // rejected
            safe_insert(operators,removed_ops.second); //remove annihilator
            safe_insert(operators,removed_ops.first); //remove creator
            return boost::make_tuple(1,false,time_diff,prob,valid_move_generated);
        }
    } else {// else
        return boost::make_tuple(1,false,0.0,0.0,valid_move_generated);
    }
}

template<typename SCALAR, typename R, typename M_TYPE, typename SLIDING_WINDOW>
boost::tuple<bool,double,bool,int>
shift_lazy(R & rng, SCALAR & det, double BETA, operator_container_t & creation_operators,
      operator_container_t & annihilation_operators, M_TYPE & M, SCALAR &sign, SCALAR &trace,
      operator_container_t & operators, double distance,
      SLIDING_WINDOW& sliding_window)
{
    namespace bll = boost::lambda;
    typedef operator_container_t::iterator it_t;

    std::vector<double> trace_bound(sliding_window.get_num_brakets());

    assert(distance<=BETA);
    bool accepted = false;

    if (creation_operators.size() == 0) {
        return boost::make_tuple(accepted,0.0,false,-1);
    }

    // shift creation operator (type=0) or annihilation operator (type=1)
    const double tau_low = sliding_window.get_tau_low();
    const double tau_high = sliding_window.get_tau_high();

    OPERATOR_TYPE type = (rng() < 0.5 ? CREATION_OP : ANNIHILATION_OP);

    operator_container_t::iterator it;
    int position;
    if (type == 0) {//shift a creation operator
        std::pair<it_t,it_t> range = creation_operators.range(tau_low<=bll::_1, bll::_1<=tau_high);
        int num_ops = std::distance(range.first,range.second);
        if (num_ops==0) {
            return boost::make_tuple(accepted,0.0,false,-1);
        }
        position = std::distance(creation_operators.begin(),range.first)+(int) (rng()*num_ops);
        it = creation_operators.begin();
        advance(it, position);
    } else {//shift an annihilation operator
        std::pair<it_t,it_t> range = annihilation_operators.range(tau_low<=bll::_1, bll::_1<=tau_high);
        int num_ops = std::distance(range.first,range.second);
        if (num_ops==0) {
            return boost::make_tuple(accepted,0.0,false,-1);
        }
        position = std::distance(annihilation_operators.begin(),range.first)+(int) (rng()*num_ops);
        it = annihilation_operators.begin();
        advance(it, position);
    }

    const int flavor = it->flavor();
    const double old_t = it->time();
    const double new_t = old_t + 2*(rng()-0.5) * distance;
    const double op_distance = std::abs(old_t-new_t);
    const psi removed_op(*it);
    const psi new_operator(new_t, type, flavor);
    if (sliding_window.get_tau_low()>new_t || new_t>sliding_window.get_tau_high() || operators.find(new_operator)!=operators.end()) {
        return boost::make_tuple(accepted,0.0,false,flavor);
    }
    assert (sliding_window.get_tau_low()<=old_t && old_t<=sliding_window.get_tau_high());
    assert (sliding_window.get_tau_low()<=new_t && new_t<=sliding_window.get_tau_high());

    double time_min, time_max;
    if (old_t < new_t) {
        time_min = old_t;
        time_max = new_t;
    } else {
        time_min = new_t;
        time_max = old_t;
    }

    operator_container_t::iterator it_op = operators.begin();

    //The following implementation is ugly, and should be simplified. I keep this as is because I am lazy (H.Shinaoka).
    int op_number = 0;            // to compute new_permutation_sign. This counts the number of operators in [time_min, time_max).

    while (it_op->time() < time_min) {
        it_op++;
    }

    if (it_op->time() == old_t) {
        it_op++;
    }

    while (it_op != operators.end() && it_op->time() < time_max) {
        op_number++;
        it_op++;
    }

    const double permutation_change = (op_number % 2 == 1 ? -1. : 1.);

    safe_erase(operators,removed_op);
    safe_insert(operators,new_operator);
    assert(removed_op.flavor()==flavor);
    assert(removed_op.type()==type);
    assert(removed_op.time()==old_t);

    const double trace_bound_sum = sliding_window.compute_trace_bound(operators, trace_bound);
    if (trace_bound_sum==0.0) {
        safe_erase(operators,new_operator);
        safe_insert(operators,removed_op);
        return boost::make_tuple(accepted,op_distance,true,flavor);
    }

    SCALAR det_rat = (type == 0 ?
                      det_rat_shift_start<SCALAR,M_TYPE,typename SLIDING_WINDOW::IMPURITY_MODEL::hybridization_container_t>
                        (new_t, position, flavor, M, annihilation_operators, sliding_window.get_p_model()->get_F(), BETA) :
                      det_rat_shift_end<SCALAR,M_TYPE,typename SLIDING_WINDOW::IMPURITY_MODEL::hybridization_container_t>
                        (new_t, position, flavor, M, creation_operators, sliding_window.get_p_model()->get_F(), BETA)
    );

    SCALAR trace_new, prob;
    const double r_th = rng();
    if (det_rat!=0.0) {
        const SCALAR rest = det_rat * permutation_change;
        const double trace_cutoff = std::abs(r_th*trace/rest);
        boost::tie(accepted,trace_new) = sliding_window.lazy_eval_trace(operators, trace_cutoff, trace_bound);
        prob = rest*(trace_new/trace);
        assert(accepted==std::abs(prob)>r_th);
    } else {
        accepted = false;
        trace_new = 0.0;
        prob = 0.0;
    }

    if (accepted) {
        int num_row_or_column_swaps;

        if (type == 0) {
            safe_erase(creation_operators,removed_op);
            safe_insert(creation_operators,new_operator);

            int new_position = 0;
            for (operator_container_t::iterator it_tmp = creation_operators.begin(); it_tmp != creation_operators.end(); ++it_tmp) {
                if (it_tmp->time()<new_t) {
                    ++new_position;
                }
            }
            num_row_or_column_swaps = compute_M_shift_start(new_t, position, new_position, flavor, M, annihilation_operators, sliding_window.get_p_model()->get_F(), BETA, det_rat);
        } else {
            safe_erase(annihilation_operators,removed_op);
            safe_insert(annihilation_operators,new_operator);

            int new_position = 0;
            for (operator_container_t::iterator it_tmp = annihilation_operators.begin(); it_tmp != annihilation_operators.end(); ++it_tmp) {
                if (it_tmp->time()<new_t) {
                    ++new_position;
                }
            }
            num_row_or_column_swaps = compute_M_shift_end(new_t, position, new_position, flavor, M, creation_operators, sliding_window.get_p_model()->get_F(), BETA, det_rat);
        }

        sign *= prob/std::abs(prob);
        trace = trace_new;
        //Note that det_rat is computed without taking into account exchanges of rows and columns in the matrix. This yields a sign flip.
        if (num_row_or_column_swaps % 2 == 1) {
            det *= -det_rat;
        } else {
            det *= det_rat;
        }
#ifndef NDEBUG
        if (!equal_det(det, 1.0/M.determinant())) {
            std::cout << "ERROR IN SHIFT UPDATE "
            << "det (fast update) = " << det
            << "det (M^-1) = " << 1./M.determinant()
            << std::endl;
            exit(1);
        }
#endif
    } else {
        safe_erase(operators,new_operator);
        safe_insert(operators,removed_op);
    }
    return boost::make_tuple(accepted,op_distance,true,flavor);
}

/**
 * @brief Try to shift the positions of all operators (in imaginary time) by random step size.
 *
 * This update is always accepted if the impurity model is translationally invariant in imaginary time.
 * If you introduce a cutoff in outer states of the trace, it may not be always the case.
 * This update will prevent Monte Carlo dynamics from getting stuck in a local minimum in such cases.
 */
template<typename SCALAR, typename R, typename M_TYPE, typename SLIDING_WINDOW>
bool
global_shift(R & rng, SCALAR & det, double BETA,  operator_container_t & creation_operators, operator_container_t & annihilation_operators,
           std::vector<int> & order_creation_flavor, std::vector<int> & order_annihilation_flavor,
           M_TYPE & M, SCALAR &sign, SCALAR &trace,
           operator_container_t & operators, SLIDING_WINDOW& sliding_window
          )
{
    assert(sliding_window.get_tau_low()==0);
    assert(sliding_window.get_tau_high()==BETA);

    if (creation_operators.size() == 0) {
        return true;
    }

    const double shift = rng()*BETA;

    //shift operators actually
    operator_container_t operators_new, creation_operators_new, annihilation_operators_new;
    global_shift_ops(creation_operators, creation_operators_new, BETA, shift);
    global_shift_ops(annihilation_operators, annihilation_operators_new, BETA, shift);
    const int num_ops_crossed = global_shift_ops(operators, operators_new, BETA, shift);

    assert(creation_operators_new.size()+annihilation_operators_new.size()==operators_new.size());

    //compute new trace
    SCALAR trace_new = sliding_window.compute_trace(operators_new);
    if (trace_new == 0.0) {
        return false;
    }

    //update inverse matrix and compute determinant ratio
    M_TYPE M_new;
    const SCALAR det_rat = update_inverse_matrix_global_shift(M, M_new, creation_operators, annihilation_operators, BETA, shift);

    const double perm_trace_change =  ( (creation_operators.size()*num_ops_crossed)%2==0 ? 1 : -1);
    const SCALAR prob = det_rat*(trace_new/trace)*perm_trace_change;

    if (rng() < std::abs(prob)) {
        sign *= prob/std::abs(prob);
        trace = trace_new;
        det *= det_rat;
        std::swap(operators, operators_new);
        std::swap(creation_operators, creation_operators_new);
        std::swap(annihilation_operators, annihilation_operators_new);
        std::swap(M,M_new);
        return true;
    } else {
#ifndef NDEBUG
        std::cerr << "global_shift: prob= " << std::abs(prob) << std::endl;
#endif
    	return false;
    }
}

template<typename SCALAR, typename R, typename M_TYPE, typename SLIDING_WINDOW, typename Iterator>
bool
exchange_flavors(R & rng, SCALAR & det, double BETA, operator_container_t & creation_operators, operator_container_t & annihilation_operators,
                 std::vector<int> & order_creation_flavor, std::vector<int> & order_annihilation_flavor,
                 M_TYPE & M, SCALAR &sign, SCALAR &trace,
                 operator_container_t & operators,
                 const SLIDING_WINDOW& sliding_window,
                 int num_flavors, Iterator new_flavors_first
) {
    assert(sliding_window.get_tau_low() == 0);
    assert(sliding_window.get_tau_high() == BETA);
    if (creation_operators.size() == 0) {
        return false;
    }

    //compute new trace
    operator_container_t operators_new;
    copy_exchange_flavors_ops(operators, operators_new, new_flavors_first);
    const SCALAR trace_new = sliding_window.compute_trace(operators_new);
    if (trace_new == 0.0) {
        return false;
    }

    //compute determinant ratio
    operator_container_t creation_operators_new, annihilation_operators_new;
    copy_exchange_flavors_ops(creation_operators, creation_operators_new, new_flavors_first);
    copy_exchange_flavors_ops(annihilation_operators, annihilation_operators_new, new_flavors_first);
    M_TYPE M_new;
    const SCALAR det_new = cal_det(creation_operators_new, annihilation_operators_new, M_new, BETA,
                                   sliding_window.get_p_model()->get_F());

    const bool isnan_tmp = std::isnan(get_real(det_new)) && std::isnan(get_imag(det_new));
    if (isnan_tmp) {
        std::cerr << "Warning: determinant of a new configuration is NaN. This may be because BETA is too large (overflow in computing determinant).";
    }

    const SCALAR prob = (det_new / det) * (trace_new / trace);//Note: no permutation sign change
    if (!isnan_tmp && rng() < std::abs(prob)) {
        sign *= prob / std::abs(prob);
        trace = trace_new;
        det = det_new;
        std::swap(operators, operators_new);
        std::swap(creation_operators, creation_operators_new);
        std::swap(annihilation_operators, annihilation_operators_new);
        std::swap(M, M_new);

        std::vector<int> order_creation_flavor_new(num_flavors), order_annihilation_flavor_new(num_flavors);
        for (int flavor=0; flavor<num_flavors; ++flavor) {
            const int new_flavor = *(new_flavors_first+flavor);
            order_creation_flavor_new[new_flavor] = order_creation_flavor[flavor];
            order_annihilation_flavor_new[new_flavor] = order_annihilation_flavor[flavor];
        }
        std::swap(order_creation_flavor, order_creation_flavor_new);
        std::swap(order_annihilation_flavor, order_annihilation_flavor_new);
        return true;
    } else {
        return false;
    }
}
