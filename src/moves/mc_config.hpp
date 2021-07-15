#pragma once

#include <typeinfo>

#include <alps/fastupdate/determinant_matrix_partitioned.hpp>

#include "../common/wide_scalar.hpp"
#include "../model/operator.hpp"
#include "../model/atomic_model.hpp"
#include "../model/hybridization_function.hpp"
#include "../sliding_window/sliding_window.hpp"

#include "worm.hpp"

template<typename SCALAR>
struct MonteCarloConfiguration {
  typedef typename ExtendedScalar<SCALAR>::value_type EXTENDED_SCALAR;
  typedef alps::fastupdate::DeterminantMatrixPartitioned<SCALAR, HybridizationFunction<SCALAR>, psi, psi>
      DeterminantMatrixType;


  MonteCarloConfiguration(boost::shared_ptr<HybridizationFunction<SCALAR> > F) :
      sign(1.0),
      trace(std::numeric_limits<double>::max()),
      M(F),
      //operators(),
      perm_sign(1)
  {
  }

  ConfigSpace current_config_space() const {
    if (!p_worm) {
      return Z_FUNCTION;
    } else {
      return p_worm->get_config_space();
    }
  }

  //int num_config_spaces() const { return static_cast<int>(CONFIG_SPACE_END); }

  void check_nan() const {
    if (my_isnan(trace)) {
      throw std::runtime_error("trace is NaN!");
    }
    if (my_isnan(sign)) {
      throw std::runtime_error("sign is NaN!");
    }
  }

  int pert_order() const {
    return M.size();
  }

  template<typename SW>
  void sanity_check(SW &sliding_window);

  SCALAR sign; // the sign of w=Z_k_up*Z_k'_down*trace
  EXTENDED_SCALAR trace;  // matrix trace
  DeterminantMatrixType M;

  // All c and c^dagger operators hybridized with bath and those from the worm
  //operator_container_t operators;

  boost::shared_ptr<Worm> p_worm;
  int perm_sign;
};

template<typename SCALAR>
template<typename SW>
void MonteCarloConfiguration<SCALAR>::sanity_check(SW &sliding_window) {
#ifndef NDEBUG
  ////operator_container_t operators2;
  //operators2.insert(M.get_cdagg_ops().begin(), M.get_cdagg_ops().end());
  //operators2.insert(M.get_c_ops().begin(), M.get_c_ops().end());
  //if (p_worm) {
    //std::vector<psi> worm_ops = p_worm->get_operators();
    //operators2.insert(worm_ops.begin(), worm_ops.end());
  //}
  //if (operators2 != operators) {
    //std::cout << "debug1 size " << operators.size() << std::endl;
    //std::cout << "debug2 size " << operators2.size() << std::endl;
    //std::cout << "debug1 " << operators << std::endl;
    //std::cout << "debug2 " << operators2 << std::endl;
    //throw std::runtime_error("operators is wrong!");
  //}
  //assert(operators2 == operators);

  //check determinant
  std::vector<SCALAR> det_old = M.compute_determinant_as_product();
  M.rebuild_inverse_matrix();
  std::vector<SCALAR> det_new = M.compute_determinant_as_product();
  SCALAR det_rat = 1.0;
  for (int i = 0; i < det_new.size(); ++i) {
    det_rat = det_old[i] / det_new[i];
  }
  assert(std::abs(det_rat - 1.0) < 1E-4);

  //check trace
  //const int Nwin = std::max(sliding_window.get_n_window(), 10);
  //typename SW::state_t state = sliding_window.get_state();
  //sliding_window.set_window_size(Nwin, operators, 0, ITIME_LEFT);
//
  //std::vector<EXTENDED_REAL> trace_bound(sliding_window.get_num_brakets());
  //sliding_window.compute_trace_bound(operators, trace_bound);
//
  //std::pair<bool, EXTENDED_SCALAR> r = sliding_window.lazy_eval_trace(operators, EXTENDED_REAL(0.0), trace_bound);
  //const EXTENDED_SCALAR trace_recomputed = r.second;
//
  //sliding_window.restore_state(operators, state);

  //assert(myabs(trace_recomputed - trace) < 1E-4 * myabs(trace));

  const int perm_sign2 = compute_permutation_sign(*this);
  assert(perm_sign2 == perm_sign);

  SCALAR sign_det = 1.0;
  for (int i = 0; i < det_new.size(); ++i) {
    sign_det *= mysign(det_new[i]);
  }
  //const SCALAR sign2 = sign_det * convert_to_scalar(mysign(trace_recomputed)) * (1. * perm_sign2);

  //assert(std::abs(sign2 / sign - 1.0) < 1E-4);
#endif
}

//compute the permutation sign (+/-) from the time-ordering of
// c_0 c^dagger_0 c^dagger_1 c_1 ... c_{N-1} c^dagger_{N-1} W_0 W_1 ...
// where creation and annihilation operators are already time-ordered, respectively.
// W_0, W_1 are operators of the worm.
template<typename SCALAR>
int compute_permutation_sign(
    const MonteCarloConfiguration<SCALAR> &mc_config
) {

  const std::vector<psi> &worm_ops = mc_config.p_worm ? mc_config.p_worm->get_operators() : std::vector<psi>();
  return compute_permutation_sign_impl(mc_config.M.get_cdagg_ops(),
                                       mc_config.M.get_c_ops(),
                                       worm_ops
  );
}

inline int compute_permutation_sign_impl(
    const std::vector<psi>& cdagg_ops,
    const std::vector<psi>& c_ops,
    const std::vector<psi>& worm_ops
) {
  std::vector<OperatorTime> times_work, work1, work2;
  const int pert_order = cdagg_ops.size();

  //Check the ordering of Cdagg, C
  work1.resize(pert_order);
  work2.resize(pert_order);
  for (int iop = 0; iop < pert_order; ++iop) {
    work1[iop] = cdagg_ops[iop].time();
    work2[iop] = c_ops[iop].time();
  }
  std::sort(work1.begin(), work1.end(), std::greater<OperatorTime>());
  std::sort(work2.begin(), work2.end(), std::greater<OperatorTime>());
  times_work.resize(0);
  for (int p = 0; p < pert_order; ++p) {
    times_work.push_back(work2[p]);
    times_work.push_back(work1[p]);
  }
  for (std::vector<psi>::const_iterator it = worm_ops.begin(); it != worm_ops.end(); ++it) {
    times_work.push_back(it->time());
  }
  const int perm_sign = alps::fastupdate::comb_sort(
      times_work.begin(), times_work.end(),
      std::greater<OperatorTime>());
  return perm_sign;
}

//The number of operators whose times are less than t.
template<typename S, typename T>
int num_ops_less_than(const S &ops, const T &t) {
  return std::distance(ops.begin(), ops.upper_bound(t));
}

//Count the number of exchanges in time ordering
//between the operators of a worm and operators hybridized with the batCount the number of exchanges between the operators of a worm and operators hybridized with the bath
template<typename S>
long count_worm_op_exchange(const S &ops,
                            const std::vector<psi> &worm_ops) {
  long count = 0;
  for (int iop = 0; iop < worm_ops.size(); ++iop) {
    count += num_ops_less_than(ops, worm_ops[iop]);
  }
  return count;
}


// Compute the change of the permutation sign (+/-) from the time-ordering of
// c^dagger_0 c_0  c^dagger_1 c_1 ... c^dagger_N c_N,
// where creation and annihilation operators are already time-ordered, respectively.
template<typename SCALAR>
int compute_permutation_sign_change(
    const operator_container_t &cdagg_ops_old,
    const operator_container_t &c_ops_old,
    const std::vector<psi> &cdagg_ops_rem,
    const std::vector<psi> &c_ops_rem,
    const std::vector<psi> &cdagg_ops_add,
    const std::vector<psi> &c_ops_add
) {
  namespace bll = boost::lambda;

  std::set<OperatorTime> cdagg_time_changed, c_time_changed;

  long count_exchange = 0;
  for (auto it = cdagg_ops_rem.begin(); it != cdagg_ops_rem.end(); ++it) {
    count_exchange += num_ops_less_than(c_ops_old, it->time());
    count_exchange += num_ops_less_than(c_time_changed, it->time());
    cdagg_time_changed.insert(it->time());
  }

  for (auto it = c_ops_rem.begin(); it != c_ops_rem.end(); ++it) {
    count_exchange += num_ops_less_than(cdagg_ops_old, it->time());
    count_exchange += num_ops_less_than(cdagg_time_changed, it->time());
    c_time_changed.insert(it->time());
  }

  for (auto it = cdagg_ops_add.begin(); it != cdagg_ops_add.end(); ++it) {
    count_exchange += num_ops_less_than(c_ops_old, it->time());
    count_exchange += num_ops_less_than(c_time_changed, it->time());
    cdagg_time_changed.insert(it->time());
  }

  for (auto it = c_ops_add.begin(); it != c_ops_add.end(); ++it) {
    count_exchange += num_ops_less_than(cdagg_ops_old, it->time());
    count_exchange += num_ops_less_than(cdagg_time_changed, it->time());
    c_time_changed.insert(it->time());
  }

  return count_exchange % 2 == 0 ? 1 : -1;
}

template<typename SCALAR>
std::vector<int>
count_creation_operators(int num_flavors, const MonteCarloConfiguration<SCALAR> &mc_config) {
  std::vector<int> num_cdagg(num_flavors, 0);
  const std::vector<psi> &creation_operators = mc_config.M.get_cdagg_ops();
  for (std::vector<psi>::const_iterator it = creation_operators.begin(); it != creation_operators.end(); ++it) {
    ++num_cdagg[it->flavor()];
  }
  return num_cdagg;
}
