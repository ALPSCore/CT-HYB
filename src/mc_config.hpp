#pragma once

#include <typeinfo>

#include <alps/fastupdate/determinant_matrix_partitioned.hpp>

#include "operator.hpp"
#include "model.hpp"
#include "wide_scalar.hpp"
#include "sliding_window.hpp"
#include "worm.hpp"

template<typename SCALAR>
class HybridizationFunction {
 private:
  typedef boost::multi_array<SCALAR, 3> container_t;

 public:
  HybridizationFunction(double BETA, int n_tau, int n_flavors, const container_t &F, double eps = 1e-10) :
      BETA_(BETA),
      F_(F),
      n_tau_(n_tau),
      n_flavors_(n_flavors),
      connected_(boost::extents[n_flavors_][n_flavors_]) {
    assert(F_[0][0].size() == n_tau + 1);
    for (int flavor = 0; flavor < n_flavors; ++flavor) {
      for (int flavor2 = 0; flavor2 < n_flavors; ++flavor2) {
        connected_[flavor][flavor2] = false;
        for (int itau = 0; itau < n_tau + 1; ++itau) {
          if (std::abs(F[flavor][flavor2][itau]) > eps) {
            connected_[flavor][flavor2] = true;
          }
        }
      }
    }
  }

  int num_flavors() const { return n_flavors_; }

  SCALAR operator()(const psi &c_op, const psi &cdagger_op) const {
    double sign = 1;
    double t = c_op.time() - cdagger_op.time();
    if (t < 0) {
      t += BETA_;
      sign = -1;
    }

    double n = t / BETA_ * n_tau_;
    int n_lower = (int) n;
    const SCALAR *pF = &F_[c_op.flavor()][cdagger_op.flavor()][0];
    return sign * (pF[n_lower] + (n - n_lower) * (pF[n_lower + 1] - pF[n_lower]));
  }

  bool is_connected(int flavor1, int flavor2) const {
    return connected_[flavor1][flavor2];
  }

 private:
  double BETA_;
  int n_tau_, n_flavors_;
  container_t F_;
  boost::multi_array<bool, 2> connected_;
};

enum CONFIG_SPACE {
  Z_FUNCTION_SPACE,
  N2_SPACE,
  CONFIG_SPACE_END
};

template<typename SCALAR>
struct MonteCarloConfiguration {
  typedef typename ExtendedScalar<SCALAR>::value_type EXTENDED_SCALAR;
  typedef alps::fastupdate::DeterminantMatrixPartitioned<SCALAR, HybridizationFunction<SCALAR>, psi, psi>
      DeterminantMatrixType;


  MonteCarloConfiguration(boost::shared_ptr<HybridizationFunction<SCALAR> > F) :
      sign(1.0),
      trace(std::numeric_limits<double>::max()),
      M(F),
      operators(),
      perm_sign(1) {
  }

  CONFIG_SPACE current_config_space() const {
    typedef CorrelationWorm<2> N2Worm;
    if (p_worm) {
      return Z_FUNCTION_SPACE;
    }
    if (typeid(*p_worm.get()) == typeid(N2Worm)) {
      return N2_SPACE;
    }
    throw std::runtime_error("Unknown worm type!");
  }

  int num_config_spaces() const { return static_cast<int>(CONFIG_SPACE_END); }

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

  SCALAR sign;                            // the sign of w=Z_k_up*Z_k'_down*trace
  EXTENDED_SCALAR trace;        // matrix trace
  DeterminantMatrixType M;
  operator_container_t operators; //all c and c^dagger operators hybridized with bath and those from the worm
  boost::shared_ptr<Worm> p_worm;
  int perm_sign;
};

template<typename SCALAR>
template<typename SW>
void MonteCarloConfiguration<SCALAR>::sanity_check(SW &sliding_window) {
#ifndef NDEBUG
  operator_container_t operators2;
  operators2.insert(M.get_cdagg_ops().begin(), M.get_cdagg_ops().end());
  operators2.insert(M.get_c_ops().begin(), M.get_c_ops().end());
  if (operators2 != operators) {
    std::cout << "debug1 size " << operators.size() << std::endl;
    std::cout << "debug2 size " << operators2.size() << std::endl;
    std::cout << "debug1 " << operators << std::endl;
    std::cout << "debug2 " << operators2 << std::endl;
  }
  assert(operators2 == operators);

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
  const int Nwin = std::max(sliding_window.get_n_window(), 10);
  typename SW::state_t state = sliding_window.get_state();
  sliding_window.set_window_size(Nwin, operators, 0, ITIME_LEFT);

  std::vector<EXTENDED_REAL> trace_bound(sliding_window.get_num_brakets());
  sliding_window.compute_trace_bound(operators, trace_bound);

  std::pair<bool, EXTENDED_SCALAR> r = sliding_window.lazy_eval_trace(operators, EXTENDED_REAL(0.0), trace_bound);
  const EXTENDED_SCALAR trace_recomputed = r.second;

  sliding_window.restore_state(operators, state);

  assert(myabs(trace_recomputed - trace) < 1E-4 * myabs(trace));

  const int perm_sign2 = compute_permutation_sign(*this);
  assert(perm_sign2 == perm_sign);

  SCALAR sign_det = 1.0;
  for (int i = 0; i < det_new.size(); ++i) {
    sign_det *= mysign(det_new[i]);
  }
  const SCALAR sign2 = sign_det * convert_to_scalar(mysign(trace_recomputed)) * (1. * perm_sign2);

  assert(std::abs(sign2 / sign - 1.0) < 1E-4);
#endif
}

//compute the permutation sign (+/-) from the time-ordering of
// c^dagger_0 c_0  c^dagger_1 c_1 ... c^dagger_N c_N  W_0 W_1 ...
// where creation and annihilation operators are already time-ordered, respectively.
// W_0, W_1 are operators of the worm.
template<typename SCALAR>
int compute_permutation_sign(
    const MonteCarloConfiguration<SCALAR> &mc_config
) {
  typedef operator_container_t::const_iterator IteratorType;

  std::vector<OperatorTime> times_work, work1, work2;

  //Check the ordering of Cdagg, C
  work1.resize(mc_config.pert_order());
  work2.resize(mc_config.pert_order());
  for (int iop = 0; iop < mc_config.pert_order(); ++iop) {
    work1[iop] = mc_config.M.get_cdagg_ops()[iop].time();
    work2[iop] = mc_config.M.get_c_ops()[iop].time();
  }
  std::sort(work1.begin(), work1.end(), OperatorTimeLessor());
  std::sort(work2.begin(), work2.end(), OperatorTimeLessor());
  times_work.resize(0);
  for (int pert_order = 0; pert_order < mc_config.pert_order(); ++pert_order) {
    times_work.push_back(work1[pert_order]);
    times_work.push_back(work2[pert_order]);
  }
  if (mc_config.p_worm) {
    const std::vector<psi> &worm_ops = mc_config.p_worm->get_operators();
    for (std::vector<psi>::const_iterator it = worm_ops.cbegin(); it != worm_ops.cend(); ++it) {
      times_work.push_back(it->time());
    }
  }
  const int perm_sign = alps::fastupdate::comb_sort(
      times_work.begin(), times_work.end(),
      OperatorTimeLessor());
  return perm_sign;
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
