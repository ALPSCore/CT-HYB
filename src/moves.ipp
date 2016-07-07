#include "moves.hpp"

/**
 * @brief pick one of elements randombly
 */
template<typename T>
inline
const T &pick(const std::vector<T> &array, alps::random01 &rng) {
  return array[static_cast<int>(rng() * array.size())];
}

/**
 * @brief pick a n elements randombly from 0, 1, ..., N-1
 */
template<class R>
std::vector<int> pickup_a_few_numbers(int N, int n, R &random01) {
  std::vector<int> flag(N, 0), list(n);

  for (int i = 0; i < n; ++i) {
    int itmp = 0;
    while (true) {
      itmp = static_cast<int>(random01() * N);
      if (flag[itmp] == 0) {
        break;
      }
    }
    list[i] = itmp;
    flag[itmp] = 1;
  }
  return list;
}

inline void range_check(std::vector<psi> &ops, double tau_low, double tau_high) {
  for (std::vector<psi>::iterator it = ops.begin(); it != ops.end(); ++it) {
    if (tau_low > it->time().time() || tau_high < it->time().time()) {
      throw std::runtime_error("Something went wrong: try to update operators outside the range");
    }
  }
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::update(
    alps::random01 &rng, double BETA,
    MonteCarloConfiguration<SCALAR> &mc_config,
    SLIDING_WINDOW &sliding_window
) {
  accepted_ = false;

  valid_move_generated_ = propose(
      rng, mc_config, sliding_window
  );

  if (!valid_move_generated_) {
    finalize_update();
    return;
  } else {
    assert(acceptance_rate_correction_);
  }

  //make sure all operators to be updated in the window
  range_check(cdagg_ops_rem_, sliding_window.get_tau_low(), sliding_window.get_tau_high());
  range_check(c_ops_rem_, sliding_window.get_tau_low(), sliding_window.get_tau_high());
  range_check(cdagg_ops_add_, sliding_window.get_tau_low(), sliding_window.get_tau_high());
  range_check(c_ops_add_, sliding_window.get_tau_low(), sliding_window.get_tau_high());

  //update operators
  bool update_success = update_operators(mc_config);
  if (!update_success) {
    finalize_update();
    return;
  }

  //compute the upper bound of trace
  trace_bound.resize(sliding_window.get_num_brakets());
  const EXTENDED_REAL trace_bound_sum = sliding_window.compute_trace_bound(mc_config.operators, trace_bound);
  if (trace_bound_sum == 0.0) {
    revert_operators(mc_config);
    finalize_update();
    return;
  }

  //compute the determinant ratio
  const SCALAR det_rat =
      mc_config.M.try_update(
          cdagg_ops_rem_.begin(), cdagg_ops_rem_.end(),
          c_ops_rem_.begin(), c_ops_rem_.end(),
          cdagg_ops_add_.begin(), cdagg_ops_add_.end(),
          c_ops_add_.begin(), c_ops_add_.end()
      );

  //flip a coin for lazy evaluation of trace
  const double r_th = rng();

  bool accepted = false;
  EXTENDED_SCALAR trace_new;
  SCALAR prob;
  if (det_rat != 0.0) {
    const SCALAR rest = (*acceptance_rate_correction_) * det_rat;
    const EXTENDED_REAL trace_cutoff = myabs(r_th * mc_config.trace / rest);
    boost::tie(accepted, trace_new) = sliding_window.lazy_eval_trace(mc_config.operators, trace_cutoff, trace_bound);
    prob = rest * convert_to_scalar(static_cast<EXTENDED_SCALAR>(trace_new / mc_config.trace));
    assert(myabs(trace_new) < myabs(trace_bound_sum) * 1.01);
    assert(accepted == std::abs(prob) > r_th);
  } else {
    trace_new = 0.0;
    prob = 0.0;
  }

  if (accepted) { // move accepted
    mc_config.M.perform_update();
    const int perm_new = compute_permutation_sign(mc_config);
    mc_config.sign *= (1. * perm_new / mc_config.perm_sign) * mysign(prob);
    assert(!my_isnan(mc_config.sign));
    mc_config.perm_sign = perm_new;
    mc_config.trace = trace_new;
    accepted_ = true;
  } else { // rejected
    mc_config.M.reject_update();
    revert_operators(mc_config);
  }
  mc_config.check_nan();

  finalize_update();
};

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool LocalUpdater<SCALAR,
                  EXTENDED_SCALAR,
                  SLIDING_WINDOW>::update_operators(MonteCarloConfiguration<SCALAR> &mc_config) {
  //erase should success. Otherwise, something went wrong in the proposal of the update.
  safe_erase(mc_config.operators, cdagg_ops_rem_.begin(), cdagg_ops_rem_.end());
  safe_erase(mc_config.operators, c_ops_rem_.begin(), c_ops_rem_.end());

  //insertion is more delicate. We may try to add two operators at the same tau accidentally.
  bool duplicate_found = false;
  {
    duplicate_check_work_.resize(0);
    std::copy(cdagg_ops_add_.begin(), cdagg_ops_add_.end(), std::back_inserter(duplicate_check_work_));
    std::copy(c_ops_add_.begin(), c_ops_add_.end(), std::back_inserter(duplicate_check_work_));
    std::sort(duplicate_check_work_.begin(), duplicate_check_work_.end());
    if (boost::adjacent_find(duplicate_check_work_) != duplicate_check_work_.end()) {
      duplicate_found = true;
    } else {
      for (std::vector<psi>::iterator it = duplicate_check_work_.begin(); it != duplicate_check_work_.end(); ++it) {
        if (mc_config.operators.find(*it) != mc_config.operators.end()) {
          duplicate_found = true;
          break;
        }
      }
    }
  }
  if (duplicate_found) {
    std::cout << "duplicate found " << std::endl;
    try {
      safe_insert(mc_config.operators, cdagg_ops_rem_.begin(), cdagg_ops_rem_.end());
      safe_insert(mc_config.operators, c_ops_rem_.begin(), c_ops_rem_.end());
    } catch (std::exception &e) {
      throw std::runtime_error("Insertion error B");
    }
    return false;
  }

  try {
    safe_insert(mc_config.operators, cdagg_ops_add_.begin(), cdagg_ops_add_.end());
    safe_insert(mc_config.operators, c_ops_add_.begin(), c_ops_add_.end());
  } catch (std::exception &e) {
    throw std::runtime_error("Insertion error A");
  }
  return true;
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void LocalUpdater<SCALAR,
                  EXTENDED_SCALAR,
                  SLIDING_WINDOW>::revert_operators(MonteCarloConfiguration<SCALAR> &mc_config) {
  safe_erase(mc_config.operators, cdagg_ops_add_.begin(), cdagg_ops_add_.end());
  safe_erase(mc_config.operators, c_ops_add_.begin(), c_ops_add_.end());
  try {
    safe_insert(mc_config.operators, cdagg_ops_rem_.begin(), cdagg_ops_rem_.end());
    safe_insert(mc_config.operators, c_ops_rem_.begin(), c_ops_rem_.end());
  } catch (std::exception &e) {
    throw std::runtime_error("Insertion error C");
  }
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::finalize_update() {
  call_back();

  acceptance_rate_correction_ = boost::none;
  cdagg_ops_rem_.resize(0);
  c_ops_rem_.resize(0);
  cdagg_ops_add_.resize(0);
  c_ops_add_.resize(0);
  valid_move_generated_ = false;
  accepted_ = false;
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool InsertionRemovalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window
) {
  namespace bll = boost::lambda;

  tau_low_ = sliding_window.get_tau_low();
  tau_high_ = sliding_window.get_tau_high();

  const int num_blocks = mc_config.M.num_blocks();
  num_cdagg_ops_in_range_.resize(num_blocks);
  num_c_ops_in_range_.resize(num_blocks);
  cdagg_ops_range_.resize(num_blocks);
  c_ops_range_.resize(num_blocks);
  for (int ib = 0; ib < num_blocks; ++ib) {
    cdagg_ops_range_[ib] = mc_config.M.get_cdagg_ops_set(ib).range(
        tau_low_ <= bll::_1, bll::_1 <= tau_high_
    );
    c_ops_range_[ib] = mc_config.M.get_c_ops_set(ib).range(
        tau_low_ <= bll::_1, bll::_1 <= tau_high_
    );
    num_cdagg_ops_in_range_[ib] = std::distance(cdagg_ops_range_[ib].first, cdagg_ops_range_[ib].second);
    num_c_ops_in_range_[ib] = std::distance(c_ops_range_[ib].first, c_ops_range_[ib].second);
  }

  if (rng() < 0.5) {
    return propose_insertion(rng, mc_config);
  } else {
    return propose_removal(rng, mc_config);
  }
}


template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool InsertionRemovalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose_insertion(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config) {
  namespace bll = boost::lambda;
  typedef operator_container_t::iterator it_t;

  const int num_blocks = mc_config.M.num_blocks();
  std::vector<int> num_new_pairs(num_blocks, 0);
  for (int iop = 0; iop < update_rank_; ++iop) {
    const int block = static_cast<int>(rng() * num_blocks);
    num_new_pairs[block] += 1;
    BaseType::cdagg_ops_add_.push_back(
        psi(open_random(rng, tau_low_, tau_high_),
            CREATION_OP,
            pick(mc_config.M.flavors(block), rng)
        )
    );
    BaseType::c_ops_add_.push_back(
        psi(open_random(rng, tau_low_, tau_high_),
            ANNIHILATION_OP,
            pick(mc_config.M.flavors(block), rng)
        )
    );
  }

  BaseType::acceptance_rate_correction_ = std::pow(tau_high_ - tau_low_, 2. * update_rank_);
  for (int ib = 0; ib < num_blocks; ++ib) {
    if (num_new_pairs[ib] == 0) {
      continue;
    }
    BaseType::acceptance_rate_correction_ =
        (*BaseType::acceptance_rate_correction_) *
            boost::math::factorial<double>(1. * num_new_pairs[ib])
            * (1. * mc_config.M.num_flavors(ib))
            * (1. * mc_config.M.num_flavors(ib));
    BaseType::acceptance_rate_correction_ =
        (*BaseType::acceptance_rate_correction_) /
            boost::math::binomial_coefficient<double>(
                num_cdagg_ops_in_range_[ib] + num_new_pairs[ib],
                num_new_pairs[ib]
            );
    BaseType::acceptance_rate_correction_ =
        (*BaseType::acceptance_rate_correction_) /
            boost::math::binomial_coefficient<double>(
                num_c_ops_in_range_[ib] + num_new_pairs[ib],
                num_new_pairs[ib]
            );
  }

  return true;
};

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool InsertionRemovalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose_removal(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config) {

  const int num_blocks = mc_config.M.num_blocks();
  std::vector<int> num_pairs_rem(num_blocks);
  for (int iop = 0; iop < update_rank_; ++iop) {
    num_pairs_rem[static_cast<int>(rng() * num_blocks)] += 1;
  }

  //check if there are enough operators to be removed
  for (int ib = 0; ib < num_blocks; ++ib) {
    if (num_cdagg_ops_in_range_[ib] < num_pairs_rem[ib] || num_c_ops_in_range_[ib] < num_pairs_rem[ib]) {
      return false;
    }
  }

  //pick up operators
  for (int ib = 0; ib < num_blocks; ++ib) {
    const std::vector<int> &idx_c = pickup_a_few_numbers(num_cdagg_ops_in_range_[ib], num_pairs_rem[ib], rng);
    const std::vector<int> &idx_a = pickup_a_few_numbers(num_c_ops_in_range_[ib], num_pairs_rem[ib], rng);
    for (int iop = 0; iop < num_pairs_rem[ib]; ++iop) {
      {
        it_t it = cdagg_ops_range_[ib].first;
        std::advance(it, idx_c[iop]);
        BaseType::cdagg_ops_rem_.push_back(*it);
      }
      {
        it_t it = c_ops_range_[ib].first;
        std::advance(it, idx_a[iop]);
        BaseType::c_ops_rem_.push_back(*it);
      }
    }
  }

  BaseType::acceptance_rate_correction_ = 1.0 / std::pow(tau_high_ - tau_low_, 2. * update_rank_);
  for (int ib = 0; ib < num_blocks; ++ib) {
    if (num_pairs_rem[ib] == 0) {
      continue;
    }
    BaseType::acceptance_rate_correction_ =
        (*BaseType::acceptance_rate_correction_) /
            (
                boost::math::factorial<double>(1. * num_pairs_rem[ib])
                    * (1. * mc_config.M.num_flavors(ib))
                    * (1. * mc_config.M.num_flavors(ib))
            );
    BaseType::acceptance_rate_correction_ =
        (*BaseType::acceptance_rate_correction_) *
            boost::math::binomial_coefficient<double>(num_cdagg_ops_in_range_[ib], num_pairs_rem[ib]);
    BaseType::acceptance_rate_correction_ =
        (*BaseType::acceptance_rate_correction_) *
            boost::math::binomial_coefficient<double>(num_c_ops_in_range_[ib], num_pairs_rem[ib]);
  }

  return true;
};

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool InsertionRemovalDiagonalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window
) {
  namespace bll = boost::lambda;
  typedef operator_container_t::iterator IteratorType;

  tau_low_ = sliding_window.get_tau_low();
  tau_high_ = sliding_window.get_tau_high();

  flavor_ = static_cast<int>(rng() * num_flavors_);
  const int block = mc_config.M.block_belonging_to(flavor_);

  //count creation operators
  {
    std::pair<IteratorType, IteratorType> ops_range = mc_config.M.get_cdagg_ops_set(block).range(
        tau_low_ <= bll::_1, bll::_1 <= tau_high_
    );
    cdagg_ops_in_range_.resize(0);
    for (IteratorType it = ops_range.first; it != ops_range.second; ++it) {
      if (it->flavor() == flavor_) {
        cdagg_ops_in_range_.push_back(*it);
      }
    }
  }

  //count annihilation operators
  {
    c_ops_in_range_.resize(0);
    std::pair<IteratorType, IteratorType> ops_range = mc_config.M.get_c_ops_set(block).range(
        tau_low_ <= bll::_1, bll::_1 <= tau_high_
    );
    for (IteratorType it = ops_range.first; it != ops_range.second; ++it) {
      if (it->flavor() == flavor_) {
        c_ops_in_range_.push_back(*it);
      }
    }
  }

  if (rng() < 0.5) {
    for (int iop = 0; iop < update_rank_; ++iop) {
      BaseType::cdagg_ops_add_.push_back(
          psi(open_random(rng, tau_low_, tau_high_),
              CREATION_OP,
              flavor_)
      );
      BaseType::c_ops_add_.push_back(
          psi(open_random(rng, tau_low_, tau_high_),
              ANNIHILATION_OP,
              flavor_)
      );
    }
    BaseType::acceptance_rate_correction_ =
        std::pow(tau_high_ - tau_low_, 2. * update_rank_) / (
            boost::math::binomial_coefficient<double>(cdagg_ops_in_range_.size() + update_rank_, update_rank_)
                * boost::math::binomial_coefficient<double>(c_ops_in_range_.size() + update_rank_, update_rank_)
                * boost::math::factorial<double>(update_rank_)
        );

    std::sort(BaseType::cdagg_ops_add_.begin(), BaseType::cdagg_ops_add_.end());
    std::sort(BaseType::c_ops_add_.begin(), BaseType::c_ops_add_.end());
    distance_ = std::max(BaseType::cdagg_ops_add_.back().time().time(), BaseType::c_ops_add_.back().time().time())
        - std::min(BaseType::cdagg_ops_add_[0].time().time(), BaseType::c_ops_add_[0].time().time());
    return true;
  } else {
    if (cdagg_ops_in_range_.size() < update_rank_ || c_ops_in_range_.size() < update_rank_) {
      return false;
    }
    const std::vector<int> &idx_c = pickup_a_few_numbers(cdagg_ops_in_range_.size(), update_rank_, rng);
    const std::vector<int> &idx_a = pickup_a_few_numbers(c_ops_in_range_.size(), update_rank_, rng);
    for (int iop = 0; iop < update_rank_; ++iop) {
      BaseType::cdagg_ops_rem_.push_back(
          cdagg_ops_in_range_[idx_c[iop]]
      );
      BaseType::c_ops_rem_.push_back(
          c_ops_in_range_[idx_a[iop]]
      );
    }
    BaseType::acceptance_rate_correction_ =
        (boost::math::binomial_coefficient<double>(cdagg_ops_in_range_.size(), update_rank_)
            * boost::math::binomial_coefficient<double>(c_ops_in_range_.size(), update_rank_)
            * boost::math::factorial<double>(update_rank_)
        )
            / std::pow(tau_high_ - tau_low_, 2. * update_rank_);
    std::sort(BaseType::cdagg_ops_rem_.begin(), BaseType::cdagg_ops_rem_.end());
    std::sort(BaseType::c_ops_rem_.begin(), BaseType::c_ops_rem_.end());
    distance_ = std::max(BaseType::cdagg_ops_rem_.back().time().time(), BaseType::c_ops_rem_.back().time().time())
        - std::min(BaseType::cdagg_ops_rem_[0].time().time(), BaseType::c_ops_rem_[0].time().time());
    return true;
  }
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void InsertionRemovalDiagonalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::call_back() {
  if (!BaseType::valid_move_generated_) {
    return;
  }

  if (BaseType::accepted_) {
    acc_rate_.add_sample(std::min(distance_, beta_ - distance_), 1.0, flavor_);
  } else {
    acc_rate_.add_sample(std::min(distance_, beta_ - distance_), 0.0, flavor_);
  }
};

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void InsertionRemovalDiagonalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::
create_measurement_acc_rate(alps::accumulators::accumulator_set &measurements) {
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >(
      "InsertionRemovalDiagonalRank"
          + boost::lexical_cast<std::string>(update_rank_)
          + "_attempted");

  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >(
      "InsertionRemovalDiagonalRank"
          + boost::lexical_cast<std::string>(update_rank_)
          + "_accepted");
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void InsertionRemovalDiagonalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::
measure_acc_rate(alps::accumulators::accumulator_set &measurements) {
  measurements[
      "InsertionRemovalDiagonalRank"
          + boost::lexical_cast<std::string>(update_rank_)
          + "_attempted"
  ] << to_std_vector(acc_rate_.get_counter());
  measurements["InsertionRemovalDiagonalRank"
      + boost::lexical_cast<std::string>(update_rank_)
      + "_accepted"] << to_std_vector(acc_rate_.get_sumval());
  acc_rate_.reset();
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool SingleOperatorShiftUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window) {
  namespace bll = boost::lambda;
  typedef operator_container_t::iterator IteratorType;

  double tau_low = sliding_window.get_tau_low();
  double tau_high = sliding_window.get_tau_high();

  std::pair<IteratorType, IteratorType> cdagg_ops_range =
      mc_config.operators.range(tau_low <= bll::_1, bll::_1 <= tau_high);
  const int num_cdagg_ops = std::distance(cdagg_ops_range.first, cdagg_ops_range.second);

  std::pair<IteratorType, IteratorType> c_ops_range =
      mc_config.operators.range(tau_low <= bll::_1, bll::_1 <= tau_high);
  const int num_c_ops = std::distance(c_ops_range.first, c_ops_range.second);

  if (num_cdagg_ops + num_c_ops == 0) {
    return false;
  }

  const int idx = static_cast<int>(rng() * (num_cdagg_ops + num_c_ops));

  if (idx < num_cdagg_ops) {
    IteratorType it = cdagg_ops_range.first;
    std::advance(it, idx);
    flavor_ = it->flavor();
    const int new_flavor = rng() < 0.5 ? it->flavor() : gen_new_flavor(mc_config, flavor_, rng);
    const double new_time = (2 * rng() - 1.0) * max_distance_[flavor_] + it->time().time();
    if (new_time < tau_low || new_time > tau_high) { return false; }
    distance_ = std::abs(it->time().time() - new_time);
    BaseType::cdagg_ops_rem_.push_back(*it);
    BaseType::cdagg_ops_add_.push_back(
        psi(new_time, CREATION_OP, new_flavor)
    );
  } else {
    IteratorType it = c_ops_range.first;
    std::advance(it, idx - num_cdagg_ops);
    flavor_ = it->flavor();
    const int new_flavor = rng() < 0.5 ? it->flavor() : gen_new_flavor(mc_config, flavor_, rng);
    const double new_time = (2 * rng() - 1.0) * max_distance_[flavor_] + it->time().time();
    if (new_time < tau_low || new_time > tau_high) { return false; }
    distance_ = std::abs(it->time().time() - new_time);
    BaseType::c_ops_rem_.push_back(*it);
    BaseType::c_ops_add_.push_back(
        psi(new_time, ANNIHILATION_OP, new_flavor)
    );
  }
  BaseType::acceptance_rate_correction_ = 1.0;
  return true;
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void SingleOperatorShiftUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::call_back() {
  if (!BaseType::valid_move_generated_) {
    return;
  }

  if (BaseType::accepted_) {
    acc_rate_.add_sample(distance_, 1.0, flavor_);
  } else {
    acc_rate_.add_sample(distance_, 0.0, flavor_);
  }
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void SingleOperatorShiftUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::update_parameters() {
  for (int flavor = 0; flavor < num_flavors_; ++flavor) {
    max_distance_[flavor] = acc_rate_.update_cutoff(1E-2, 1.05);
  }
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void SingleOperatorShiftUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::
create_measurement_acc_rate(alps::accumulators::accumulator_set &measurements) {
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("Shift_attempted");
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("Shift_accepted");
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void SingleOperatorShiftUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::
measure_acc_rate(alps::accumulators::accumulator_set &measurements) {
  measurements["Shift_attempted"] << to_std_vector(acc_rate_.get_counter());
  measurements["Shift_accepted"] << to_std_vector(acc_rate_.get_sumval());
  acc_rate_.reset();
}


