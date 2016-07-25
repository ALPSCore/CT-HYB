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

inline void range_check(const std::vector<psi> &ops, double tau_low, double tau_high) {
  for (std::vector<psi>::const_iterator it = ops.begin(); it != ops.end(); ++it) {
    if (tau_low > it->time().time() || tau_high < it->time().time()) {
      throw std::runtime_error("Something went wrong: try to update operators outside the range");
    }
  }
}

template<typename T>
struct InRange {
  InRange(double tau_low, double tau_high) : tau_low_(tau_low), tau_high_(tau_high) { };
  bool operator()(const T &t) const {
    return tau_low_ <= t && t <= tau_high_;
  }
  double tau_low_, tau_high_;
};

template<typename T>
struct OutOfRange {
  OutOfRange(double tau_low, double tau_high) : tau_low_(tau_low), tau_high_(tau_high) { };
  bool operator()(const T &t) const {
    return !(tau_low_ <= t && t <= tau_high_);
  }
  double tau_low_, tau_high_;
};

inline void take_worm_diff(const Worm &worm_old, const Worm &worm_new, double tau_low, double tau_high,
                    std::vector<psi> &worm_ops_rem, std::vector<psi> &worm_ops_add) {
  const OutOfRange<psi> out_of_range = OutOfRange<psi>(tau_low, tau_high);
  std::vector<psi> worm_ops_old = worm_old.get_operators();
  std::vector<psi> worm_ops_new = worm_new.get_operators();
  worm_ops_add.resize(0);
  worm_ops_rem.resize(0);
  std::remove_copy_if(worm_ops_new.begin(), worm_ops_new.end(), std::back_inserter(worm_ops_add), out_of_range);
  std::remove_copy_if(worm_ops_old.begin(), worm_ops_old.end(), std::back_inserter(worm_ops_rem), out_of_range);
  assert(worm_ops_add.size() == worm_ops_rem.size());

  //make sure operators out side the rage is not going to be modified.
  for (std::vector<psi>::const_iterator it = worm_ops_old.begin(); it != worm_ops_old.end(); ++it) {
    if (!out_of_range(*it)) {
      continue;
    }
    if (std::find(worm_ops_new.begin(), worm_ops_new.end(), *it) == worm_ops_new.end()) {
      throw std::runtime_error("Error in take_worm_diff: you are going to modify an operator outside the window");
    }
  }
  for (std::vector<psi>::const_iterator it = worm_ops_new.begin(); it != worm_ops_new.end(); ++it) {
    if (!out_of_range(*it)) {
      continue;
    }
    if (std::find(worm_ops_old.begin(), worm_ops_old.end(), *it) == worm_ops_old.end()) {
      throw std::runtime_error("Error in take_worm_diff: you are going to modify an operator outside the window");
    }
  }
}


template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::update(
    alps::random01 &rng, double BETA,
    MonteCarloConfiguration<SCALAR> &mc_config,
    SLIDING_WINDOW &sliding_window
) {
  mc_config.sanity_check(sliding_window);

  accepted_ = false;

  //by default, the state of the worm is not updated.
  p_new_worm_ = mc_config.p_worm;

  valid_move_generated_ = propose(
      rng, mc_config, sliding_window
  );

  if (!valid_move_generated_) {
    finalize_update();
    return;
  } else {
    assert(acceptance_rate_correction_);
  }

  const double tau_low = sliding_window.get_tau_low();
  const double tau_high = sliding_window.get_tau_high();

  std::vector<psi> worm_ops_rem, worm_ops_add;
  if (!mc_config.p_worm && p_new_worm_) {
    //worm insertion
    worm_ops_add = p_new_worm_->get_operators();
  } else if (mc_config.p_worm && !p_new_worm_) {
    //worm removal
  } else if (mc_config.p_worm && !p_new_worm_) {
    worm_ops_rem = mc_config.p_worm->get_operators();
  } else if (mc_config.p_worm && p_new_worm_ && *mc_config.p_worm != *p_new_worm_) {
    //worm move
    take_worm_diff(*mc_config.p_worm, *p_new_worm_, tau_low, tau_high, worm_ops_rem, worm_ops_add);
  }

  //make sure all operators to be updated in the window
  range_check(cdagg_ops_rem_, tau_low, tau_high);
  range_check(c_ops_rem_, tau_low, tau_high);
  range_check(cdagg_ops_add_, tau_low, tau_high);
  range_check(c_ops_add_, tau_low, tau_high);
  range_check(worm_ops_add, tau_low, tau_high);
  range_check(worm_ops_rem, tau_low, tau_high);

  //update operators hybirized with bath and those from worms
  bool update_success = update_operators(mc_config, worm_ops_rem, worm_ops_add);
  if (!update_success) {
    finalize_update();
    return;
  }

  //compute the upper bound of trace
  trace_bound.resize(sliding_window.get_num_brakets());
  const EXTENDED_REAL trace_bound_sum = sliding_window.compute_trace_bound(mc_config.operators, trace_bound);
  if (trace_bound_sum == 0.0) {
    revert_operators(mc_config, worm_ops_rem, worm_ops_add);
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
    mc_config.trace = trace_new;
    mc_config.p_worm = p_new_worm_;
    const int perm_new = compute_permutation_sign(mc_config);
    mc_config.sign *= (1. * perm_new / mc_config.perm_sign) * mysign(prob);
    mc_config.perm_sign = perm_new;
    assert(!my_isnan(mc_config.sign));
    accepted_ = true;
  } else { // rejected
    mc_config.M.reject_update();
    revert_operators(mc_config, worm_ops_rem, worm_ops_add);
  }
  mc_config.check_nan();

  finalize_update();
};

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool LocalUpdater<SCALAR,
                  EXTENDED_SCALAR,
                  SLIDING_WINDOW>::update_operators(MonteCarloConfiguration<SCALAR> &mc_config,
                                                    const std::vector<psi> &worm_ops_rem, const std::vector<psi> &worm_ops_add) {
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
    if (boost::adjacent_find(duplicate_check_work_, OperatorEqualTime()) != duplicate_check_work_.end()) {
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
  } catch (std::exception &e) {
    throw std::runtime_error("Insertion error A.1");
  }
  try {
    safe_insert(mc_config.operators, c_ops_add_.begin(), c_ops_add_.end());
  } catch (std::exception &e) {
    throw std::runtime_error("Insertion error A.2");
  }

  try {
    if (!mc_config.p_worm && p_new_worm_) {
      safe_insert(mc_config.operators, p_new_worm_->get_operators());
    } else if (mc_config.p_worm && !p_new_worm_) {
      safe_erase(mc_config.operators, mc_config.p_worm->get_operators());
    } else if (mc_config.p_worm && p_new_worm_ && *mc_config.p_worm != *p_new_worm_) {
      safe_erase(mc_config.operators, worm_ops_rem);
      safe_insert(mc_config.operators, worm_ops_add);
    }
  } catch (std::exception &e) {
    throw std::runtime_error("Insertion error E");
  }

  return true;
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void LocalUpdater<SCALAR,
                  EXTENDED_SCALAR,
                  SLIDING_WINDOW>::revert_operators(MonteCarloConfiguration<SCALAR> &mc_config,
                                                    const std::vector<psi> &worm_ops_rem, const std::vector<psi> &worm_ops_add) {
  safe_erase(mc_config.operators, cdagg_ops_add_.begin(), cdagg_ops_add_.end());
  safe_erase(mc_config.operators, c_ops_add_.begin(), c_ops_add_.end());
  try {
    safe_insert(mc_config.operators, cdagg_ops_rem_.begin(), cdagg_ops_rem_.end());
    safe_insert(mc_config.operators, c_ops_rem_.begin(), c_ops_rem_.end());
  } catch (std::exception &e) {
    throw std::runtime_error("Insertion error C");
  }

  try {
    if (!mc_config.p_worm && p_new_worm_) {
      safe_erase(mc_config.operators, p_new_worm_->get_operators());
    } else if (mc_config.p_worm && !p_new_worm_) {
      safe_insert(mc_config.operators, mc_config.p_worm->get_operators());
    } else if (mc_config.p_worm && p_new_worm_ && *mc_config.p_worm != *p_new_worm_) {
      safe_erase(mc_config.operators, worm_ops_add);
      safe_insert(mc_config.operators, worm_ops_rem);
    }
  } catch (std::exception &e) {
    throw std::runtime_error("Insertion error F");
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
  p_new_worm_.reset();
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
bool OperatorPairFlavorUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window
) {
  namespace bll = boost::lambda;
  typedef operator_container_t::iterator it_t;

  const double tau_low = sliding_window.get_tau_low();
  const double tau_high = sliding_window.get_tau_high();

  std::pair<it_t, it_t> cdagg_range = mc_config.M.get_cdagg_ops_set().range(tau_low <= bll::_1, bll::_1 <= tau_high);
  std::pair<it_t, it_t> c_range = mc_config.M.get_c_ops_set().range(tau_low <= bll::_1, bll::_1 <= tau_high);

  const int num_cdagg_ops = std::distance(cdagg_range.first, cdagg_range.second);
  const int num_c_ops = std::distance(c_range.first, c_range.second);

  if (num_cdagg_ops == 0 || num_c_ops == 0) {
    return false;
  }

  //asign a new random flavor to one of creation operators
  it_t it_cdagg = cdagg_range.first;
  std::advance(it_cdagg, static_cast<int>(num_cdagg_ops*rng()));
  BaseType::cdagg_ops_rem_.push_back(*it_cdagg);
  psi cdagg_op_new = *it_cdagg;
  cdagg_op_new.set_flavor(static_cast<int>(num_flavors_*rng()));
  BaseType::cdagg_ops_add_.push_back(cdagg_op_new);

  //asign a new random flavor to one of annihilation operators
  it_t it_c = c_range.first;
  std::advance(it_c, static_cast<int>(num_c_ops*rng()));
  BaseType::c_ops_rem_.push_back(*it_c);
  psi c_op_new = *it_c;
  c_op_new.set_flavor(static_cast<int>(num_flavors_*rng()));
  BaseType::c_ops_add_.push_back(c_op_new);

  BaseType::acceptance_rate_correction_ = 1.0;

  return true;
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
      mc_config.M.get_cdagg_ops_set().range(tau_low <= bll::_1, bll::_1 <= tau_high);
  const int num_cdagg_ops = std::distance(cdagg_ops_range.first, cdagg_ops_range.second);

  std::pair<IteratorType, IteratorType> c_ops_range =
      mc_config.M.get_c_ops_set().range(tau_low <= bll::_1, bll::_1 <= tau_high);
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

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
int SingleOperatorShiftUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::
gen_new_flavor(const MonteCarloConfiguration<SCALAR> &mc_config, int old_flavor, alps::random01 &rng) {
  const int block = mc_config.M.block_belonging_to(old_flavor);
  return pick(mc_config.M.flavors(block), rng);
}

template<typename SCALAR>
SCALAR compute_det_rat(
    const std::vector<SCALAR> &det_vec_new,
    const std::vector<SCALAR> &det_vec_old,
    double eps = 1e-30) {
  const int num_loop = std::max(det_vec_new.size(), det_vec_old.size());

  const double max_abs_elem = std::abs(*std::max_element(det_vec_new.begin(), det_vec_new.end(), AbsLessor<SCALAR>()));

  SCALAR det_rat = 1.0;
  for (int i = 0; i < num_loop; ++i) {
    if (i < det_vec_new.size() && std::abs(det_vec_new[i] / max_abs_elem) > eps) {
      det_rat *= det_vec_new[i];
    }
    if (i < det_vec_old.size()) {
      det_rat /= det_vec_old[i];
    }
  }
  return det_rat;
}

template<typename Scalar, typename M>
std::vector<Scalar>
lu_product(const M &matrix) {
  if (matrix.rows() == 0) {
    return std::vector<Scalar>();
  }
  Eigen::FullPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > lu(matrix);
  const int size1 = lu.rows();
  std::vector<Scalar> results(size1);
  for (int i = 0; i < size1; ++i) {
    results[i] = lu.matrixLU()(i, i);
  }
  results[0] *= lu.permutationP().determinant() * lu.permutationQ().determinant();
  return results;
};

template<typename SCALAR, typename GreensFunction, typename DetMatType>
SCALAR compute_det_rat(const std::vector<psi> &creation_operators,
                       const std::vector<psi> &annihilation_operators,
                       std::vector<SCALAR> &det_vec_old,
                       DetMatType &M,
                       std::vector<SCALAR> &det_vec_new
) {
  typedef alps::fastupdate::ResizableMatrix<SCALAR> M_TYPE;
  std::vector<std::vector<psi> > cdagg_ops(M.num_blocks()), c_ops(M.num_blocks());

  for (std::vector<psi>::const_iterator it = creation_operators.begin(); it != creation_operators.end(); ++it) {
    cdagg_ops[M.block_belonging_to(it->flavor())].push_back(*it);
  }
  for (std::vector<psi>::const_iterator it = annihilation_operators.begin(); it != annihilation_operators.end(); ++it) {
    c_ops[M.block_belonging_to(it->flavor())].push_back(*it);
  }


  //compute determinant as a product
  boost::shared_ptr<GreensFunction> p_gf = M.get_greens_function();
  std::vector<OperatorTime> cdagg_times, c_times;
  det_vec_new.resize(0);
  det_vec_new.reserve(creation_operators.size());
  for (int ib = 0; ib < M.num_blocks(); ++ib) {
    const int mat_size = cdagg_ops[ib].size();
    assert(cdagg_ops[ib].size() == c_ops[ib].size());
    if (mat_size == 0) {
      continue;
    }
    Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> M_new(mat_size, mat_size);
    for (int col = 0; col < mat_size; ++col) {
      for (int row = 0; row < mat_size; ++row) {
        M_new(row, col) = p_gf->operator()(c_ops[ib][row], cdagg_ops[ib][col]);
      }
    }
    const std::vector<SCALAR> &vec_tmp = lu_product<SCALAR>(M_new);
    std::copy(vec_tmp.begin(), vec_tmp.end(), std::back_inserter(det_vec_new));

    for (int col = 0; col < mat_size; ++col) {
      cdagg_times.push_back(cdagg_ops[ib][col].time());
    }
    for (int row = 0; row < mat_size; ++row) {
      c_times.push_back(c_ops[ib][row].time());
    }
  }

  if (det_vec_new.size() == 0) {
    return 0.0;
  }

  //compute determinant ratio
  std::sort(det_vec_old.begin(), det_vec_old.end(), AbsGreater<SCALAR>());
  std::sort(det_vec_new.begin(), det_vec_new.end(), AbsGreater<SCALAR>());
  const SCALAR det_rat = compute_det_rat(det_vec_new, det_vec_old);

  //compute permulation sign from exchange of row and col
  const int perm_sign_block = alps::fastupdate::comb_sort(cdagg_times.begin(), cdagg_times.end(), OperatorTimeLessor())
      * alps::fastupdate::comb_sort(c_times.begin(), c_times.end(), OperatorTimeLessor());

  det_vec_new[0] *= 1. * perm_sign_block;
  return (1. * perm_sign_block) * det_rat;
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename R, typename SLIDING_WINDOW,
    typename HybridizedOperatorTransformer, typename WormTransformer>
bool
global_update(R &rng,
              double BETA,
              MonteCarloConfiguration<SCALAR> &mc_config,
              std::vector<SCALAR> &det_vec,
              SLIDING_WINDOW &sliding_window,
              int num_flavors,
              const HybridizedOperatorTransformer &hyb_op_transformer,
              const WormTransformer &worm_transformer,
              int Nwin
) {
  assert(sliding_window.get_tau_low() == 0);
  assert(sliding_window.get_tau_high() == BETA);
  mc_config.sanity_check(sliding_window);
  const int pert_order = mc_config.pert_order();
  if (pert_order == 0) {
    return true;
  }

  //compute new operators hybridized with bath
  std::vector<psi> creation_operators_new, annihilation_operators_new;
  std::transform(
      mc_config.M.get_cdagg_ops().begin(), mc_config.M.get_cdagg_ops().end(),
      std::back_inserter(creation_operators_new),
      hyb_op_transformer);
  std::transform(
      mc_config.M.get_c_ops().begin(), mc_config.M.get_c_ops().end(),
      std::back_inserter(annihilation_operators_new),
      hyb_op_transformer);


  //create operator list
  operator_container_t operators_new;
  operators_new.insert(creation_operators_new.begin(), creation_operators_new.end());
  operators_new.insert(annihilation_operators_new.begin(), annihilation_operators_new.end());
  //worm
  boost::shared_ptr<Worm> p_new_worm;
  if (mc_config.p_worm) {
    boost::shared_ptr<Worm> p_w = worm_transformer(*(mc_config.p_worm));
    p_new_worm.swap(p_w);
    std::vector<psi> new_worm_ops = p_new_worm->get_operators();
    operators_new.insert(new_worm_ops.begin(), new_worm_ops.end());
  }

  //compute new trace (we use sliding window to avoid overflow/underflow).
  sliding_window.set_window_size(1, mc_config.operators, 0, ITIME_LEFT);
  sliding_window.set_window_size(Nwin, operators_new, 0, ITIME_LEFT);

  std::vector<EXTENDED_REAL> trace_bound(sliding_window.get_num_brakets());
  sliding_window.compute_trace_bound(operators_new, trace_bound);

  std::pair<bool, EXTENDED_SCALAR> r = sliding_window.lazy_eval_trace(operators_new, EXTENDED_REAL(0.0), trace_bound);
  const EXTENDED_SCALAR trace_new = r.second;

  sliding_window.set_window_size(1, mc_config.operators, 0, ITIME_LEFT);
  if (trace_new == EXTENDED_SCALAR(0.0)) {
    return false;
  }

  //compute determinant ratio
  std::vector<SCALAR> det_vec_new;
  const SCALAR det_rat = compute_det_rat<SCALAR, HybridizationFunction<SCALAR> >(
      creation_operators_new, annihilation_operators_new,
      det_vec, mc_config.M, det_vec_new);

  const SCALAR prob =
      convert_to_scalar(
          EXTENDED_SCALAR(
              EXTENDED_SCALAR(det_rat) *
                  EXTENDED_SCALAR(trace_new / mc_config.trace)
          )
      );

  if (rng() < std::abs(prob)) {
    std::vector<std::pair<psi, psi> > operator_pairs(pert_order);
    for (int iop = 0; iop < pert_order; ++iop) {
      operator_pairs[iop] = std::make_pair(creation_operators_new[iop], annihilation_operators_new[iop]);
    }
    typedef typename MonteCarloConfiguration<SCALAR>::DeterminantMatrixType DeterminantMatrixType;
    DeterminantMatrixType M_new(
        mc_config.M.get_greens_function(),
        operator_pairs.begin(),
        operator_pairs.end()
    );

    mc_config.trace = trace_new;
    std::swap(mc_config.operators, operators_new);
    std::swap(mc_config.M, M_new);
    std::swap(det_vec, det_vec_new);
    if (mc_config.p_worm) {
      mc_config.p_worm.swap(p_new_worm);
    }
    const int perm_sign_new = compute_permutation_sign(mc_config);
    mc_config.sign *= (1. * perm_sign_new / mc_config.perm_sign) * prob / std::abs(prob);
    mc_config.perm_sign = perm_sign_new;
    mc_config.sanity_check(sliding_window);
    return true;
  } else {
    return false;
  }
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
WormUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::WormUpdater(
    const std::string &str, double beta, int num_flavors, double tau_lower_limit, double tau_upper_limit) :
    str_(str),
    beta_(beta),
    num_flavors_(num_flavors),
    tau_lower_limit_(tau_lower_limit),
    tau_upper_limit_(tau_upper_limit),
    acc_rate_(1000, 0.5 * beta, 1, 0.5 * beta),
    max_distance_(0.5 * beta),
    distance_(-1.0),
    worm_space_weight_(1.0) {
}


template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void WormUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::call_back() {
  if (!BaseType::valid_move_generated_) {
    return;
  }

  if (BaseType::accepted_) {
    acc_rate_.add_sample(std::min(distance_, beta_ - distance_), 1.0, 0);
  } else {
    acc_rate_.add_sample(std::min(distance_, beta_ - distance_), 0.0, 0);
  }
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void WormUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::
create_measurement_acc_rate(alps::accumulators::accumulator_set &measurements) {
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >(str_ + "_attempted");
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >(str_ + "_accepted");
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void WormUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::
measure_acc_rate(alps::accumulators::accumulator_set &measurements) {
  measurements[str_ + "_attempted"] << to_std_vector(acc_rate_.get_counter());
  measurements[str_ + "_accepted"] << to_std_vector(acc_rate_.get_sumval());
  acc_rate_.reset();
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void WormUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::update_parameters() {
  max_distance_ = acc_rate_.update_cutoff(1E-2, 1.05);
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool WormMover<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window
) {
  if (!mc_config.p_worm) {
    return false;
  }

  //count independent times in the time window
  const double tau_low = std::max(sliding_window.get_tau_low(), BaseType::tau_lower_limit_);
  const double tau_high = std::min(sliding_window.get_tau_high(), BaseType::tau_upper_limit_);
  const int num_times = mc_config.p_worm->num_independent_times();
  BaseType::p_new_worm_ = mc_config.p_worm->clone();
  BaseType::distance_ = 0.0;
  bool is_movable = false;
  for (int t = 0; t < num_times; ++t) {
    if (InRange<OperatorTime>(tau_low, tau_high)(mc_config.p_worm->get_time(t))) {
      const double new_time = (2 * rng() - 1.0) * BaseType::max_distance_ + BaseType::p_new_worm_->get_time(t);
      if (new_time < tau_low || new_time > tau_high) {
        //is_movable = false;
        return false;
      }
      BaseType::distance_ = std::max(
          BaseType::distance_,
          std::abs(new_time - BaseType::p_new_worm_->get_time(t))
      );
      BaseType::p_new_worm_->set_time(t, new_time);
      is_movable = true;
    }
  }
  if (!is_movable) {
    BaseType::p_new_worm_.reset();
    return false;
  }
  //update flavor indices
  if (rng() < 0.5) {
    for (int f = 0; f < BaseType::p_new_worm_->num_independent_flavors(); ++f) {
      bool updatable = true;
      const std::vector<int> &time_index = BaseType::p_new_worm_->get_time_index(f);
      for (int t = 0; t < time_index.size(); ++ t) {
        updatable = updatable
            && InRange<double>(tau_low, tau_high)(BaseType::p_new_worm_->get_time(time_index[t]));
      }
      if (updatable) {
        BaseType::p_new_worm_->set_flavor(f, static_cast<int>(rng() * BaseType::num_flavors_));
      }
    }
  }

  BaseType::acceptance_rate_correction_ = 1.0;
  return true;
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool WormInsertionRemover<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window
) {
  const double tau_low = std::max(sliding_window.get_tau_low(), BaseType::tau_lower_limit_);
  const double tau_high = std::min(sliding_window.get_tau_high(), BaseType::tau_upper_limit_);

  const int num_time_indices = p_worm_template_->num_independent_times();
  const int num_flavor_indices = p_worm_template_->num_independent_flavors();

  const double weight_scaled = weight_ / insertion_proposal_rate_;

  if (mc_config.p_worm) {
    //propose removal
    assert(typeid(mc_config.p_worm.get()) == typeid(p_worm_template_.get()));
    if (!is_worm_in_range(*mc_config.p_worm, tau_low, tau_high)) {
      return false;
    }
    if (rng() < 0.5) {
      //diagonal & off-diagonal in flavor
      BaseType::acceptance_rate_correction_ =
          1. / (weight_scaled *
                  std::pow(tau_high - tau_low, num_time_indices) *
                  std::pow(1. * BaseType::num_flavors_, num_flavor_indices));
    } else {
      //diagonal in flavor
      if (!is_worm_diagonal_in_flavor(*mc_config.p_worm)) {
        return false;
      }
      BaseType::acceptance_rate_correction_ =
          1. / (weight_scaled *
              std::pow(tau_high - tau_low, num_time_indices) *
              BaseType::num_flavors_);
    }
    BaseType::p_new_worm_.reset();
  } else {
    //propose insertion
    BaseType::p_new_worm_ = p_worm_template_->clone();

    for (int t = 0; t < num_time_indices; ++t) {
      BaseType::p_new_worm_->set_time(t, open_random(rng, tau_low, tau_high));
    }
    if (rng() < 0.5) {
      for (int f = 0; f < num_flavor_indices; ++f) {
        BaseType::p_new_worm_->set_flavor(f, static_cast<int>(rng() * BaseType::num_flavors_));
      }
      BaseType::acceptance_rate_correction_ = weight_scaled *
          std::pow(tau_high - tau_low, num_time_indices) *
          std::pow(1. * BaseType::num_flavors_, num_flavor_indices);
    } else {
      //diagonal in flavor
      int diagonal_flavor = static_cast<int>(rng() * BaseType::num_flavors_);
      for (int f = 0; f < num_flavor_indices; ++f) {
        BaseType::p_new_worm_->set_flavor(f, diagonal_flavor);
      }
      BaseType::acceptance_rate_correction_ = weight_scaled *
          std::pow(tau_high - tau_low, num_time_indices) * BaseType::num_flavors_;
    }
  }
  return true;
}
