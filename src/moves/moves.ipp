#include "moves.hpp"

/**
 * @brief pick one of elements randombly
 */
template<typename T>
inline
const T &pick(const std::vector<T> &array, alps::random01 &rng) {
  return array[static_cast<int>(rng() * array.size())];
}


inline void range_check(const std::vector<psi> &ops, double tau_low, double tau_high) {
  for (std::vector<psi>::const_iterator it = ops.begin(); it != ops.end(); ++it) {
    if (tau_low > it->time().time() || tau_high < it->time().time()) {
      std::cout << ops << std::endl;
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

inline void merge_diff_impl(
    std::vector<psi> &op_rem,
    std::vector<psi> &op_add,
    const std::vector<psi> &op_add_new) {

  for (std::vector<psi>::const_iterator it = op_add_new.begin(); it != op_add_new.end(); ++it) {
    std::vector<psi>::iterator it2 = std::find(op_rem.begin(), op_rem.end(), *it);
    if (it2 == op_rem.end()) {
      op_add.push_back(*it);
    } else {
      std::swap(*it2, op_rem.back());
      op_rem.pop_back();
    }
  }
}

inline void merge_diff(const std::vector<psi> &hyb_op_rem,
                       const std::vector<psi> &hyb_op_add,
                       const std::vector<psi> &worm_op_rem,
                       const std::vector<psi> &worm_op_add,
                       std::vector<psi> &op_rem,
                       std::vector<psi> &op_add) {
  op_rem.resize(0);
  op_add.resize(0);
  std::copy(hyb_op_rem.begin(), hyb_op_rem.end(), std::back_inserter(op_rem));
  std::copy(worm_op_rem.begin(), worm_op_rem.end(), std::back_inserter(op_rem));
  merge_diff_impl(op_rem, op_add, hyb_op_add);
  merge_diff_impl(op_rem, op_add, worm_op_add);
}


template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::update(
    alps::random01 &rng, double BETA,
    MonteCarloConfiguration<SCALAR> &mc_config,
    SLIDING_WINDOW &sliding_window,
    const std::map<ConfigSpace, double> &config_space_weight
) {
  mc_config.sanity_check(sliding_window);

  accepted_ = false;

  //by default, the state of the worm is not updated.
  p_new_worm_ = mc_config.p_worm;

  valid_move_generated_ = propose(
      rng, mc_config, sliding_window, config_space_weight
  );

  if (!valid_move_generated_) {
    finalize_update();
    return false;
  } else {
    assert(acceptance_rate_correction_);
  }

  const double tau_low = sliding_window.get_tau_low();
  const double tau_high = sliding_window.get_tau_high();

  //Figure out which operators are actually removed and what operators are added into the trace
  std::vector<psi> op_rem_tot, op_add_tot;
  {
    std::vector<psi> hyb_op_rem, hyb_op_add;
    hyb_op_rem.reserve(cdagg_ops_rem_.size() + c_ops_rem_.size());
    std::copy(cdagg_ops_rem_.begin(), cdagg_ops_rem_.end(), std::back_inserter(hyb_op_rem));
    std::copy(c_ops_rem_.begin(), c_ops_rem_.end(), std::back_inserter(hyb_op_rem));

    hyb_op_add.reserve(cdagg_ops_add_.size() + c_ops_add_.size());
    std::copy(cdagg_ops_add_.begin(), cdagg_ops_add_.end(), std::back_inserter(hyb_op_add));
    std::copy(c_ops_add_.begin(), c_ops_add_.end(), std::back_inserter(hyb_op_add));

    const std::vector<psi> worm_ops_old = mc_config.p_worm ? mc_config.p_worm->get_operators() : std::vector<psi>();
    const std::vector<psi> worm_ops_new = p_new_worm_ ? p_new_worm_->get_operators() : std::vector<psi>();

    merge_diff(hyb_op_rem, hyb_op_add, worm_ops_old, worm_ops_new, op_rem_tot, op_add_tot);

    range_check(op_rem_tot, tau_low, tau_high);
    range_check(op_add_tot, tau_low, tau_high);
  }

  //update operators
  std::vector<std::pair<psi, ActionType> > update_record;
  bool update_success = update_operators(mc_config.operators, op_rem_tot, op_add_tot, update_record);
  if (!update_success) {
    finalize_update();
    return false;
  }

  bool accepted = false;
  SCALAR prob;
  EXTENDED_SCALAR trace_new;
  if (op_rem_tot.size() == 0 && op_add_tot.size() == 0) {
    //if trace is not updated, just compute det_rat
    trace_new = mc_config.trace;

    //compute the determinant ratio
    const SCALAR det_rat =
        mc_config.M.try_update(
            cdagg_ops_rem_.begin(), cdagg_ops_rem_.end(),
            c_ops_rem_.begin(), c_ops_rem_.end(),
            cdagg_ops_add_.begin(), cdagg_ops_add_.end(),
            c_ops_add_.begin(), c_ops_add_.end()
        );

    prob = (*acceptance_rate_correction_) * det_rat;
    accepted = rng() < std::abs(prob);
  } else {
    //compute the upper bound of trace
    trace_bound.resize(sliding_window.get_num_brakets());
    const EXTENDED_REAL trace_bound_sum = sliding_window.compute_trace_bound(mc_config.operators, trace_bound);
    if (trace_bound_sum == 0.0) {
      //revert_operators(mc_config, worm_ops_rem, worm_ops_add);
      revert_changes(mc_config.operators, update_record);
      finalize_update();
      return false;
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
    revert_changes(mc_config.operators, update_record);
  }
  mc_config.check_nan();

  finalize_update();

  return accepted;
};

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool LocalUpdater<SCALAR,
                  EXTENDED_SCALAR,
                  SLIDING_WINDOW>::update_operators(operator_container_t &operators,
                                                    const std::vector<psi> &ops_rem,
                                                    const std::vector<psi> &ops_add,
                                                    std::vector<std::pair<psi, ActionType> > &update_record) {
  update_record.resize(0);

  //First, remove operators
  try {
    safe_erase_with_record(operators, ops_rem.begin(), ops_rem.end(), update_record);
  } catch (const std::exception &e) {
    throw std::runtime_error("Fatal error in LocalUpdater::update_operators");
  }

  //Then, add operators
  try {
    safe_insert_with_record(operators, ops_add.begin(), ops_add.end(), update_record);
  } catch (const std::exception &e) {
    //In the case, we try to insert an operator at tau where there is already another operator, we reject such an update.
    std::cout << "Warning: duplicate found" << std::endl;
    revert_changes(operators, update_record);
    return false;
  }

  return true;
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void LocalUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::finalize_update() {
  ++ num_attempted_;
  if (valid_move_generated_) {
    ++ num_valid_move_;
    if (accepted_) {
      ++ num_accepted_;
    }
  }

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
    const SLIDING_WINDOW &sliding_window,
    const std::map<ConfigSpace, double> &config_space_weight
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
        assert(tau_low_ <= it->time().time());
        assert(tau_high_ >= it->time().time());
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
    const SLIDING_WINDOW &sliding_window,
    const std::map<ConfigSpace, double> &config_space_weight
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
  BaseType::create_measurement_acc_rate(measurements);
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
  BaseType::measure_acc_rate(measurements);
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
    const SLIDING_WINDOW &sliding_window,
    const std::map<ConfigSpace, double> &config_space_weight
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
  std::advance(it_cdagg, static_cast<int>(num_cdagg_ops * rng()));
  BaseType::cdagg_ops_rem_.push_back(*it_cdagg);
  psi cdagg_op_new = *it_cdagg;
  cdagg_op_new.set_flavor(static_cast<int>(num_flavors_ * rng()));
  BaseType::cdagg_ops_add_.push_back(cdagg_op_new);

  //asign a new random flavor to one of annihilation operators
  it_t it_c = c_range.first;
  std::advance(it_c, static_cast<int>(num_c_ops * rng()));
  BaseType::c_ops_rem_.push_back(*it_c);
  psi c_op_new = *it_c;
  c_op_new.set_flavor(static_cast<int>(num_flavors_ * rng()));
  BaseType::c_ops_add_.push_back(c_op_new);

  BaseType::acceptance_rate_correction_ = 1.0;

  return true;
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool SingleOperatorShiftUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window,
    const std::map<ConfigSpace, double> &config_space_weight
) {
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
  BaseType::create_measurement_acc_rate(measurements);
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("Shift_attempted");
  measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("Shift_accepted");
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void SingleOperatorShiftUpdater<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::
measure_acc_rate(alps::accumulators::accumulator_set &measurements) {
  BaseType::measure_acc_rate(measurements);
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
    if (cdagg_ops[ib].size() != c_ops[ib].size()) {
      return 0.0;
    }
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
  const int perm_sign_block = alps::fastupdate::comb_sort(cdagg_times.begin(), cdagg_times.end(), std::less<OperatorTime>())
      * alps::fastupdate::comb_sort(c_times.begin(), c_times.end(), std::less<OperatorTime>());

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
    BaseType(str),
    str_(str),
    beta_(beta),
    num_flavors_(num_flavors),
    tau_lower_limit_(tau_lower_limit),
    tau_upper_limit_(tau_upper_limit) {
  if (tau_lower_limit != 0.0 || tau_upper_limit != beta) {
    throw std::runtime_error("Error in constructor of WormUpdater");
  }
}


template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
void WormMover<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::call_back() {
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
void WormMover<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::update_parameters() {
  max_distance_ = acc_rate_.update_cutoff(0.1, 1.05);
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool WormMover<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window,
    const std::map<ConfigSpace, double> &config_space_weight
) {
  if (!mc_config.p_worm) {
    return false;
  }

  BaseType::p_new_worm_.reset();

  //count independent times in the time window
  const double tau_low = std::max(sliding_window.get_tau_low(), BaseType::tau_lower_limit_);
  const double tau_high = std::min(sliding_window.get_tau_high(), BaseType::tau_upper_limit_);
  const int num_times = mc_config.p_worm->num_independent_times();
  distance_ = 0.0;
  for (int itry = 0; itry < mc_config.p_worm->num_independent_times(); ++ itry) {
    const int t = static_cast<int>(rng() * mc_config.p_worm->num_independent_times());
    if (!InRange<OperatorTime>(tau_low, tau_high)(mc_config.p_worm->get_time(t))) {
      continue;
    }
    const double new_time = rng() < 0.9 ?
                            (2 * rng() - 1.0) * max_distance_ + mc_config.p_worm->get_time(t) :
                            open_random(rng, tau_low, tau_high);
    if (new_time < tau_low || new_time > tau_high) {
      continue;
    }
    BaseType::p_new_worm_ = mc_config.p_worm->clone();
    const double dist_tmp = std::abs(new_time - BaseType::p_new_worm_->get_time(t));
    distance_ = std::min(dist_tmp, beta_ - dist_tmp);
    BaseType::p_new_worm_->set_time(t, new_time);
    break;
  }

  if (BaseType::p_new_worm_) {
    BaseType::acceptance_rate_correction_ = 1.0;
    return true;
  } else {
    return false;
  }
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool WormFlavorChanger<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window,
    const std::map<ConfigSpace, double> &config_space_weight
) {
  if (!mc_config.p_worm) {
    return false;
  }

  //count independent times in the time window
  const double tau_low = std::max(sliding_window.get_tau_low(), BaseType::tau_lower_limit_);
  const double tau_high = std::min(sliding_window.get_tau_high(), BaseType::tau_upper_limit_);

  BaseType::p_new_worm_ = mc_config.p_worm->clone();

  for (int f = 0; f < BaseType::p_new_worm_->num_independent_flavors(); ++f) {
    bool updatable = true;
    const std::vector<int> &time_index = BaseType::p_new_worm_->get_time_index(f);
    for (int t = 0; t < time_index.size(); ++t) {
      updatable = updatable
          && InRange<double>(tau_low, tau_high)(BaseType::p_new_worm_->get_time(time_index[t]));
    }
    if (updatable) {
      BaseType::p_new_worm_->set_flavor(f, static_cast<int>(rng() * BaseType::num_flavors_));
    }
  }

  if (*mc_config.p_worm != *BaseType::p_new_worm_) {
    BaseType::acceptance_rate_correction_ = 1.0;
    return true;
  } else {
    BaseType::p_new_worm_.reset();
    return false;
  }
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool WormInsertionRemover<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window,
    const std::map<ConfigSpace, double> &config_space_weight
) {
  if (mc_config.p_worm) {
    assert(typeid(mc_config.p_worm.get()) == typeid(p_worm_template_.get()));

    //check if all the operators of the worm in the sliding window
    const double tau_low = std::max(sliding_window.get_tau_low(), BaseType::tau_lower_limit_);
    const double tau_high = std::min(sliding_window.get_tau_high(), BaseType::tau_upper_limit_);
    if (!is_worm_in_range(*mc_config.p_worm, tau_low, tau_high)) {
      return false;
    }
  }

  if (rng() < 0.5) {
    //Insertion is done by adding operators into the trace
    return propose_by_trace_impl(rng, mc_config, sliding_window, config_space_weight);
  } else {
    //Insertion is done by removing operators hybridized with the bath and adding them into the trace
    //This respects quantum number conservation
    return propose_by_trace_hyb_impl(rng, mc_config, sliding_window, config_space_weight);
  }
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool WormInsertionRemover<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose_by_trace_impl(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window,
    const std::map<ConfigSpace, double> &config_space_weight
) {
  const double tau_low = std::max(sliding_window.get_tau_low(), BaseType::tau_lower_limit_);
  const double tau_high = std::min(sliding_window.get_tau_high(), BaseType::tau_upper_limit_);

  const int num_time_indices = p_worm_template_->num_independent_times();
  const int num_flavor_indices = p_worm_template_->num_independent_flavors();

  std::map<ConfigSpace, double>::const_iterator it = config_space_weight.find(p_worm_template_->get_config_space());
  if (it == config_space_weight.end()) {
    std::cout << get_config_space_name(p_worm_template_->get_config_space()) << std::endl;
    throw std::logic_error("Worm space weight not found");
  }
  const double worm_space_weight = it->second;

  const double weight_scaled = worm_space_weight / insertion_proposal_rate_;

  if (mc_config.p_worm) {
    //propose removal
    if (rng() < 0.5) {
      //diagonal & off-diagonal in flavor
      BaseType::acceptance_rate_correction_ =
          1. / (weight_scaled *
              std::pow(tau_high - tau_low, 1. * num_time_indices) *
              std::pow(1. * BaseType::num_flavors_, 1. * num_flavor_indices));
    } else {
      //diagonal in flavor
      if (!is_worm_diagonal_in_flavor(*mc_config.p_worm)) {
        return false;
      }
      BaseType::acceptance_rate_correction_ =
          1. / (weight_scaled *
              std::pow(tau_high - tau_low, 1. * num_time_indices) *
              BaseType::num_flavors_);
    }
    BaseType::p_new_worm_.reset();
  } else {
    //propose insertion
    BaseType::p_new_worm_ = p_worm_template_->clone();

    //std::set<double> duplicate_check;
    for (int t = 0; t < num_time_indices; ++t) {
      BaseType::p_new_worm_->set_time(t, open_random(rng, tau_low, tau_high));
    }
    if (rng() < 0.5) {
      for (int f = 0; f < num_flavor_indices; ++f) {
        BaseType::p_new_worm_->set_flavor(f, static_cast<int>(rng() * BaseType::num_flavors_));
      }
      BaseType::acceptance_rate_correction_ = weight_scaled *
          std::pow(tau_high - tau_low, 1. * num_time_indices) *
          std::pow(1. * BaseType::num_flavors_, 1. * num_flavor_indices);
    } else {
      //diagonal in flavor
      int diagonal_flavor = static_cast<int>(rng() * BaseType::num_flavors_);
      for (int f = 0; f < num_flavor_indices; ++f) {
        BaseType::p_new_worm_->set_flavor(f, diagonal_flavor);
      }
      BaseType::acceptance_rate_correction_ = weight_scaled *
          std::pow(tau_high - tau_low, 1. * num_time_indices) * BaseType::num_flavors_;
    }
  }
  return true;
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool WormInsertionRemover<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose_by_trace_hyb_impl(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window,
    const std::map<ConfigSpace, double> &config_space_weight
) {
  namespace bll = boost::lambda;
  typedef operator_container_t::iterator it_t;

  const double tau_low = std::max(sliding_window.get_tau_low(), BaseType::tau_lower_limit_);
  const double tau_high = std::min(sliding_window.get_tau_high(), BaseType::tau_upper_limit_);

  const int num_time_indices = p_worm_template_->num_independent_times();
  const int num_flavor_indices = p_worm_template_->num_independent_flavors();

  std::map<ConfigSpace, double>::const_iterator it = config_space_weight.find(p_worm_template_->get_config_space());
  if (it == config_space_weight.end()) {
    return false;
  }
  const double worm_space_weight = it->second;

  const double weight_scaled = worm_space_weight / insertion_proposal_rate_;

  const int num_flavors = BaseType::num_flavors_;

  //Range of operators in the window
  std::pair<it_t, it_t>
      cdagg_ops_range = mc_config.M.get_cdagg_ops_set().range(tau_low <= bll::_1, bll::_1 <= tau_high);
  std::pair<it_t, it_t> c_ops_range = mc_config.M.get_c_ops_set().range(tau_low <= bll::_1, bll::_1 <= tau_high);

  //count the number of operators hybridized with the bath in the window
  std::vector<int> num_cdagg_ops_in_range, num_c_ops_in_range;
  count_operators(cdagg_ops_range.first, cdagg_ops_range.second, num_flavors, num_cdagg_ops_in_range);
  count_operators(c_ops_range.first, c_ops_range.second, num_flavors, num_c_ops_in_range);

  if (mc_config.p_worm) {
    //make all the operators of the worm hybridized with the bath
    const std::vector<psi> &worm_ops = mc_config.p_worm->get_operators();
    std::vector<int> num_cdagg_ops_new(BaseType::num_flavors_, 0), num_c_ops_new(BaseType::num_flavors_, 0);
    for (std::vector<psi>::const_iterator it = worm_ops.begin(); it != worm_ops.end(); ++it) {
      if (it->type() == CREATION_OP) {
        BaseType::cdagg_ops_add_.push_back(*it);
        BaseType::cdagg_ops_add_.back().set_time(OperatorTime(open_random(rng, tau_low, tau_high)));
        ++num_cdagg_ops_new[it->flavor()];
      } else {
        BaseType::c_ops_add_.push_back(*it);
        BaseType::c_ops_add_.back().set_time(OperatorTime(open_random(rng, tau_low, tau_high)));
        ++num_c_ops_new[it->flavor()];
      }
    }

    //come from degrees of freedom in which we assign a random number to each operator from the worm.
    const double p_prop_worm_rem = 1.0 / std::pow(tau_high - tau_low, 1.*worm_ops.size());

    double p_prop_worm_ins;
    {
      p_prop_worm_ins = 1.0;
      for (int flavor = 0; flavor < num_flavors; ++flavor) {
        p_prop_worm_ins *=
            boost::math::binomial_coefficient<double>(num_cdagg_ops_in_range[flavor] + num_cdagg_ops_new[flavor],
                                                      num_cdagg_ops_new[flavor]);
        p_prop_worm_ins *= boost::math::binomial_coefficient<double>(num_c_ops_in_range[flavor] + num_c_ops_new[flavor],
                                                                     num_c_ops_new[flavor]);
      }
      p_prop_worm_ins = 1.0 / p_prop_worm_ins;
    }

    BaseType::acceptance_rate_correction_ =
        (p_prop_worm_ins / p_prop_worm_rem) * (1. / (weight_scaled *
            std::pow(tau_high - tau_low, 1. * num_time_indices) *
            std::pow(1. * BaseType::num_flavors_, 1. * num_flavor_indices)));

    BaseType::p_new_worm_.reset();
  } else {
    //insert a worm
    BaseType::p_new_worm_ = p_worm_template_->clone();

    //Flavors and times are asigned to the new worm randomly.
    for (int t = 0; t < num_time_indices; ++t) {
      BaseType::p_new_worm_->set_time(t, open_random(rng, tau_low, tau_high));
    }
    for (int f = 0; f < num_flavor_indices; ++f) {
      BaseType::p_new_worm_->set_flavor(f, static_cast<int>(rng() * num_flavors));
    }

    //Count operators in the worm, which will be removed from the bath
    const std::vector<psi> &worm_ops = BaseType::p_new_worm_->get_operators();
    std::vector<int> num_cdagg_ops_rem, num_c_ops_rem;
    count_operators(worm_ops.begin(), worm_ops.end(), num_flavors, num_cdagg_ops_rem, CREATION_OP);
    count_operators(worm_ops.begin(), worm_ops.end(), num_flavors, num_c_ops_rem, ANNIHILATION_OP);

    //Abort if there is no enough operator
    for (int flavor = 0; flavor < num_flavors; ++flavor) {
      if (num_cdagg_ops_in_range[flavor] < num_cdagg_ops_rem[flavor]
          || num_c_ops_in_range[flavor] < num_c_ops_rem[flavor]) {
        return false;
      }
    }

    //Remove some operators from the bath
    pick_up_operators(cdagg_ops_range.first, cdagg_ops_range.second, num_cdagg_ops_rem, BaseType::cdagg_ops_rem_, rng);
    pick_up_operators(c_ops_range.first, c_ops_range.second, num_c_ops_rem, BaseType::c_ops_rem_, rng);

    double p_prop_worm_ins;
    {
      p_prop_worm_ins = 1.0;
      for (int flavor = 0; flavor < num_flavors; ++flavor) {
        p_prop_worm_ins *=
            boost::math::binomial_coefficient<double>(num_cdagg_ops_in_range[flavor], num_cdagg_ops_rem[flavor]);
        p_prop_worm_ins *= boost::math::binomial_coefficient<double>(num_c_ops_in_range[flavor], num_c_ops_rem[flavor]);
      }
      p_prop_worm_ins = 1.0 / p_prop_worm_ins;
    }

    //come from degrees of freedom in which we assign a random number to each operator from the worm.
    const double p_prop_worm_rem = 1.0 / std::pow(tau_high - tau_low, 1. * worm_ops.size());

    BaseType::acceptance_rate_correction_ = (p_prop_worm_rem / p_prop_worm_ins) * weight_scaled *
        std::pow(tau_high - tau_low, 1. * num_time_indices) *
        std::pow(1. * BaseType::num_flavors_, 1. * num_flavor_indices);
  }
  return true;
}

template<typename SCALAR, int RANK, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool GWormInsertionRemover<SCALAR, RANK, EXTENDED_SCALAR, SLIDING_WINDOW>::propose(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window,
    const std::map<ConfigSpace, double> &config_space_weight
) {
  if (typeid(mc_config.p_worm.get()) != typeid(p_worm_template_.get())) {
    throw std::logic_error("Type is wrong in GWormInsertionRemover::update()");
  }

  std::map<ConfigSpace, double>::const_iterator it = config_space_weight.find(p_worm_template_->get_config_space());
  if (it == config_space_weight.end()) {
    throw std::logic_error("Worm space weight not found");
  }
  const double worm_space_weight = it->second;

  if (mc_config.p_worm) {
    //propose removal by connecting worm operators
    const std::vector<psi> &worm_ops = mc_config.p_worm->get_operators();
    for (int rank = 0; rank < RANK; ++rank) {
      BaseType::c_ops_add_.push_back(worm_ops[2 * rank]);
      BaseType::cdagg_ops_add_.push_back(worm_ops[2 * rank + 1]);
    }
    BaseType::acceptance_rate_correction_ = 1.0 /
        (worm_space_weight
            * std::pow(boost::math::binomial_coefficient<double>(mc_config.pert_order() + RANK, RANK), 2.0));
    BaseType::p_new_worm_.reset();
  } else {
    //propose insertion by cutting hybridization lines
    if (mc_config.pert_order() < RANK) {
      return false;
    }
    BaseType::p_new_worm_ = p_worm_template_->clone();

    const std::vector<int> &cdagg_op_indices = pickup_a_few_numbers(mc_config.pert_order(), RANK, rng);
    const std::vector<int> &c_op_indices = pickup_a_few_numbers(mc_config.pert_order(), RANK, rng);

    const std::vector<psi> &cdagg_ops_M = mc_config.M.get_cdagg_ops();
    const std::vector<psi> &c_ops_M = mc_config.M.get_c_ops();

    for (int rank = 0; rank < RANK; ++rank) {
      BaseType::cdagg_ops_rem_.push_back(cdagg_ops_M[cdagg_op_indices[rank]]);
      BaseType::c_ops_rem_.push_back(c_ops_M[c_op_indices[rank]]);

      BaseType::p_new_worm_->set_flavor(2 * rank, BaseType::c_ops_rem_.back().flavor());
      BaseType::p_new_worm_->set_flavor(2 * rank + 1, BaseType::cdagg_ops_rem_.back().flavor());

      BaseType::p_new_worm_->set_time(2 * rank, BaseType::c_ops_rem_.back().time().time());
      BaseType::p_new_worm_->set_time(2 * rank + 1, BaseType::cdagg_ops_rem_.back().time().time());
    }

    BaseType::acceptance_rate_correction_ =
        worm_space_weight * std::pow(boost::math::binomial_coefficient<double>(mc_config.pert_order(), RANK), 2.0);
  }
  return true;
}

inline double get_tau_first_hyb_op_larger_than(const operator_container_t &ops,
                          const psi &op,
                          double tau_high,
                          const std::vector<psi> &worm_ops,
                          boost::optional<psi> &hyb_op_lower_bound) {
  operator_container_t::iterator it = ops.lower_bound(op);
  if (it == ops.end()) {
    return tau_high;
  }
  //find the operator that has the smallest tau and does not belongs to the worm
  while (it != ops.end() && std::find(worm_ops.begin(), worm_ops.end(), *it) != worm_ops.end()) {
    ++it;
  }
  if (it == ops.end()) {
    return tau_high;
  } else {
    hyb_op_lower_bound = *it;
    return std::min(tau_high, it->time().time());
  }
}

inline double get_tau_first_hyb_op_smaller_than(const operator_container_t &ops,
                          const psi &op,
                          double tau_low,
                          const std::vector<psi> &worm_ops,
                          boost::optional<psi> &hyb_op_upper_bound) {
  namespace bll = boost::lambda;
  typedef operator_container_t::iterator it_t;

  //find the operator that has the largest tau and does not belongs to the worm
  std::pair<it_t, it_t> op_range = ops.range(0.0 <= bll::_1, bll::_1 < op.time());
  operator_container_t::iterator it = op_range.second;
  if (it == ops.end()) {
    return tau_low;
  }

  operator_container_t::iterator it_up = it;
  ++it_up;
  while (it_up != ops.begin() && std::find(worm_ops.begin(), worm_ops.end(), *it) != worm_ops.end()) {
    --it;
    --it_up;
  }
  if (it_up == ops.begin()) {
    return tau_low;
  } else {
    hyb_op_upper_bound = *it;
    return std::max(tau_low, it->time().time());
  }
}

/**
 * Count the number of pairs of cdagger and c operators hybridized with the bath (appearing in this order)
 * and return one of the pairs.
 */
template<typename InputItr>
int count_hyb_cdagg_c_op_pairs(InputItr op_begin,
                               InputItr op_end,
                               const std::vector<psi> &worm_ops,
                               alps::random01 &rng,
                               std::pair<psi, psi> &cdagg_c_pair) {
  std::vector<std::pair<psi, psi> > cdagg_c_pairs;
  if (std::distance(op_begin, op_end) < 2) {
    return 0;
  }
  {
    InputItr it_up = op_begin;
    ++it_up;
    for (InputItr it = op_begin; it_up != op_end; ++it, ++it_up) {
      if (it->type() == ANNIHILATION_OP && it_up->type() == CREATION_OP &&
          std::find(worm_ops.begin(), worm_ops.end(), *it) == worm_ops.end() &&
          std::find(worm_ops.begin(), worm_ops.end(), *it_up) == worm_ops.end()
          ) {
        cdagg_c_pairs.push_back(std::make_pair(*it_up, *it));
      }
    }
  }
  if (cdagg_c_pairs.size() > 0) {
    const int idx = static_cast<int>(rng() * cdagg_c_pairs.size());
    cdagg_c_pair = cdagg_c_pairs[idx];
  }
  return cdagg_c_pairs.size();
}

template<typename SCALAR, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool EqualTimeG1_TwoTimeG2_Connector<SCALAR, EXTENDED_SCALAR, SLIDING_WINDOW>::propose(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window,
    const std::map<ConfigSpace, double> &config_space_weight
) {
  typedef operator_container_t::iterator it_t;
  namespace bll = boost::lambda;

  if (mc_config.current_config_space() != Equal_time_G1 && mc_config.current_config_space() != Two_time_G2) {
    throw std::runtime_error("Error in EqualTimeG1_TwoTimeG2_Connector::propose()");
  }

  const double weight_G1 = config_space_weight.find(Equal_time_G1)->second;
  const double weight_G2 = config_space_weight.find(Two_time_G2)->second;

  const double tau_low = sliding_window.get_tau_low();
  const double tau_high = sliding_window.get_tau_high();

  const std::vector<psi> &worm_ops = mc_config.p_worm->get_operators();

  if (mc_config.current_config_space() == Equal_time_G1) {
    BaseType::p_new_worm_ = boost::shared_ptr<Worm>(new CorrelationWorm<2>());
    BaseType::p_new_worm_->set_time(0, mc_config.p_worm->get_time(0));
    BaseType::p_new_worm_->set_time(1, open_random(rng, tau_low, tau_high));
    for (int iop = 0; iop < 2; ++iop) {
      BaseType::p_new_worm_->set_flavor(iop, mc_config.p_worm->get_flavor(iop));
    }

    if (rng() < 0.5) {
      //Insert new operators into the trace
      for (int iop = 2; iop < 4; ++iop) {
        BaseType::p_new_worm_->set_flavor(iop, static_cast<int>(num_flavors_ * rng()));
      }

      BaseType::acceptance_rate_correction_ = (weight_G2 / weight_G1) *
          num_flavors_ * num_flavors_ * (tau_high - tau_low);
    } else {
      return false;
      //Insert new operators into the trace and remove hybridized operators
      std::pair<it_t, it_t> op_range = mc_config.operators.range(
          tau_low <= bll::_1, bll::_1 <= tau_high
      );
      std::pair<psi, psi> cdagg_c_pair;
      const int num_cdagg_c_pair =
          count_hyb_cdagg_c_op_pairs(op_range.first, op_range.second, worm_ops, rng, cdagg_c_pair);
      if (num_cdagg_c_pair == 0) {
        return false;
      }

      //remove cdagger and c operators from the determinant
      BaseType::cdagg_ops_rem_.push_back(cdagg_c_pair.first);
      BaseType::c_ops_rem_.push_back(cdagg_c_pair.second);

      BaseType::p_new_worm_->set_flavor(2, cdagg_c_pair.first.flavor());
      BaseType::p_new_worm_->set_flavor(3, cdagg_c_pair.second.flavor());

      boost::optional<psi> hyb_op_upper_bound, hyb_op_lower_bound;
      //tau_max: the time of the first operator which has a larger time than the operator pair under consideration.
      // If there is no such operator, tau_max = tau_high = upper end point of the sliding window.
      //tau_min: defined in a similar way
      double tau_max, tau_min;
      tau_max = get_tau_first_hyb_op_larger_than(mc_config.operators, cdagg_c_pair.first, tau_high, worm_ops, hyb_op_lower_bound);
      tau_min = get_tau_first_hyb_op_smaller_than(mc_config.operators, cdagg_c_pair.second, tau_low, worm_ops, hyb_op_upper_bound);
      BaseType::p_new_worm_->set_time(1, open_random(rng, tau_min, tau_max));

      BaseType::acceptance_rate_correction_ = (weight_G2 / weight_G1) * (2. * num_cdagg_c_pair) / (tau_max - tau_min);
      assert(tau_min < tau_max);
      assert(tau_low <= tau_min);
      assert(tau_max <= tau_high);
    }
  } else {
    if (mc_config.p_worm->get_time(1) < tau_low || mc_config.p_worm->get_time(1) > tau_high) {
      return false;
    }

    BaseType::p_new_worm_ = boost::shared_ptr<Worm>(new EqualTimeGWorm<1>());
    BaseType::p_new_worm_->set_time(0, mc_config.p_worm->get_time(0));
    for (int iop = 0; iop < 2; ++iop) {
      BaseType::p_new_worm_->set_flavor(iop, mc_config.p_worm->get_flavor(iop));
    }

    if (rng() < 0.5) {
      //Remove worm operators
      BaseType::acceptance_rate_correction_ = (weight_G1 / weight_G2) / (
          num_flavors_ * num_flavors_ * (tau_high - tau_low)
      );
    } else {
      return false;
      //Remove worm operators and insert operators into the determinant to cancel out changes in quantum numbers
      boost::optional<psi> hyb_op_upper_bound, hyb_op_lower_bound;
      const double tau_max = get_tau_first_hyb_op_larger_than(mc_config.operators, worm_ops[2], tau_high, worm_ops, hyb_op_lower_bound);
      const double tau_min = get_tau_first_hyb_op_smaller_than(mc_config.operators, worm_ops[3], tau_low, worm_ops, hyb_op_upper_bound);
      assert(tau_min < tau_max);
      assert(tau_low <= tau_min);
      assert(tau_max <= tau_high);

      double tau_cdagg_op = open_random(rng, tau_max, tau_min), tau_c_op = open_random(rng, tau_max, tau_min);
      if (tau_cdagg_op == tau_c_op) {
        return false;
      } else if (tau_cdagg_op < tau_c_op) {
        std::swap(tau_cdagg_op, tau_c_op);
      }

      //add cdagger and c operators into to the determinant
      BaseType::cdagg_ops_add_.push_back(psi(tau_cdagg_op, CREATION_OP, worm_ops[2].flavor()));
      BaseType::c_ops_add_.push_back(psi(tau_c_op, ANNIHILATION_OP, worm_ops[3].flavor()));

      //count the number of cdagger and c pairs after the update
      //(A)  cdagger   n   cdagger   => cdagger  cdagger c  cdagger   num_pairs_after_update = num_pairs_before_update+1
      //(B)  cdagger   n         c   => cdagger  cdagger c  c         num_pairs_after_update = num_pairs_before_update
      //(C)  cdagger   n   cdagger   => cdagger  cdagger c  cdagger   num_pairs_after_update = num_pairs_before_update+1
      //(D)  c         n   c         => c        cdagger c  c         num_pairs_after_update = num_pairs_before_update+1
      //(E)  Otherwise                                                num_pairs_after_update = num_pairs_before_update+1
      std::pair<it_t, it_t> op_range = mc_config.operators.range(
          tau_low <= bll::_1, bll::_1 <= tau_high
      );
      std::pair<psi, psi> hyb_op_pair;
      const int num_hyb_op_pairs =
          count_hyb_cdagg_c_op_pairs(op_range.first, op_range.second, worm_ops, rng, hyb_op_pair);

      int num_pairs_after_update;
      if (hyb_op_lower_bound && hyb_op_upper_bound && hyb_op_lower_bound->type() == CREATION_OP
          && hyb_op_upper_bound->type() == ANNIHILATION_OP) {
        num_pairs_after_update = num_hyb_op_pairs;
      } else {
        num_pairs_after_update = num_hyb_op_pairs + 1;
      }

      BaseType::acceptance_rate_correction_ =
          (weight_G1 / weight_G2) * (tau_max - tau_min) / (2. * num_pairs_after_update);
    }
  }
  return true;
}

template<typename SCALAR, int RANK, typename EXTENDED_SCALAR, typename SLIDING_WINDOW>
bool GWormShifter<SCALAR, RANK, EXTENDED_SCALAR, SLIDING_WINDOW>::propose(
    alps::random01 &rng,
    MonteCarloConfiguration<SCALAR> &mc_config,
    const SLIDING_WINDOW &sliding_window,
    const std::map<ConfigSpace, double> &config_space_weight
) {
  namespace bll = boost::lambda;
  typedef operator_container_t::iterator it_t;

  if (typeid(mc_config.p_worm.get()) != typeid(p_worm_template_.get())) {
    throw std::logic_error("Type is wrong in GWormShifter::update()");
  }

  const int iop_shifted = static_cast<int>(rng() * 2 * RANK);
  double tau_range_max, tau_range_min;
  if (rng() < 0.5) {
    tau_range_max = beta_;
    tau_range_min = 0.0;
  } else {
    double tau_old = mc_config.p_worm->get_time(iop_shifted);
    tau_range_max = tau_old + (beta_/mc_config.pert_order()) * num_flavors_;
    tau_range_min = tau_old - (beta_/mc_config.pert_order()) * num_flavors_;
  }

  const int block = mc_config.M.block_belonging_to(mc_config.p_worm->get_flavor(iop_shifted));
  const operator_container_t *p_ops = iop_shifted % 2 == 0 ?
    &mc_config.M.get_c_ops_set(block) :
     &mc_config.M.get_cdagg_ops_set(block);
  std::pair<it_t,it_t> ops_range = p_ops->range(
    tau_range_min <= bll::_1, bll::_1 <= tau_range_max
  );
  ops_work_.resize(0);
  std::copy(ops_range.first, ops_range.second, std::back_inserter(ops_work_));

  if (ops_work_.size() == 0) {
    return false;
  }

  BaseType::p_new_worm_ = mc_config.p_worm->clone();
  psi new_op = ops_work_[static_cast<int>(rng() * ops_work_.size())];
  BaseType::p_new_worm_->set_flavor(iop_shifted, new_op.flavor());
  BaseType::p_new_worm_->set_time(iop_shifted, new_op.time().time());

  const psi old_op = mc_config.p_worm->get_operators()[iop_shifted];
  if (iop_shifted % 2 == 0) {
    BaseType::c_ops_rem_.push_back(new_op);
    BaseType::c_ops_add_.push_back(old_op);
  } else {
    BaseType::cdagg_ops_rem_.push_back(new_op);
    BaseType::cdagg_ops_add_.push_back(old_op);
  }

  BaseType::acceptance_rate_correction_ = 1.0;
  return true;
}
