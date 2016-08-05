#include "./measurement.hpp"

template<typename SCALAR>
template<typename SlidingWindow>
void TwoTimeG2Measurement<SCALAR>::measure(MonteCarloConfiguration<SCALAR> &mc_config,
                                           alps::accumulators::accumulator_set &measurements,
                                           alps::random01 &random,
                                           SlidingWindow &sliding_window,
                                           int average_pert_order,
                                           const std::string &str) {
  typedef typename ExtendedScalar<SCALAR>::value_type EXTENDED_SCALAR;
  typedef operator_container_t::iterator Iterator;
  if (mc_config.current_config_space() != Two_time_G2) {
    return;
  }

  //Remove the left-hand-side operators
  operator_container_t ops(mc_config.operators);
  const std::vector<psi> worm_ops_original = mc_config.p_worm->get_operators();
  safe_erase(ops, worm_ops_original);

  //Generate times of the left hand operator pair c^dagger c
  //Make sure there is no duplicate
  const int num_times = std::max(20, average_pert_order);
  std::set<double> new_times;
  new_times.insert(worm_ops_original[0].time().time());
  if (new_times.size() < num_times) {
    std::set<double> duplicate_check;
    for (operator_container_t::iterator it = mc_config.operators.begin(); it != mc_config.operators.end(); ++it) {
      duplicate_check.insert(it->time().time());
    }
    while (new_times.size() < num_times) {
      double t = open_random(random, 0.0, beta_);
      if (duplicate_check.find(t) == duplicate_check.end()) {
        new_times.insert(t);
        duplicate_check.insert(t);
      }
    }
    assert(new_times.size() == num_times);
  }

  //remember the current status of the sliding window
  const typename SlidingWindow::state_t state = sliding_window.get_state();
  const int n_win = sliding_window.get_n_window();

  std::vector<EXTENDED_REAL> trace_bound(sliding_window.get_num_brakets());

  //configurations whose trace is smaller than this cutoff are ignored.
  const EXTENDED_REAL trace_cutoff = EXTENDED_REAL(1.0E-30) * myabs(mc_config.trace);

  double norm = 0.0;
  std::fill(data_.origin(), data_.origin() + data_.num_elements(), 0.0);

  //compute Monte Carlo weights of configurations with new time
  // sweep = 0: configuration with new time for the left-hand operator pair
  // sweep = 1: configuration with new time and new flavors for all worm operators
  for (int sweep = 0; sweep < 2; ++sweep) {
    std::vector<psi> worm_ops = worm_ops_original;
    if (sweep == 1) {//change flavors of all worm operators
      for (int iop = 0; iop < worm_ops.size(); ++iop) {
        worm_ops[iop].set_flavor(static_cast<int>(random() * num_flavors_));
      }
    }

    //insert worm operators which are not shifted in time
    for (int iop = 2; iop < worm_ops.size(); ++iop) {
      safe_insert(ops, worm_ops[iop]);
    }

    //reset the window and move to the right most position
    sliding_window.set_window_size(std::max(num_times, n_win), ops, 0, ITIME_LEFT);

    for (std::set<double>::iterator it = new_times.begin(); it != new_times.end(); ++it) {
      //move the window so that it contains the time
      while (*it > sliding_window.get_tau_high()) {
        sliding_window.move_window_to_next_position(ops);
      }
      assert(*it <= sliding_window.get_tau_high());
      assert(*it >= sliding_window.get_tau_low());

      worm_ops[0].set_time(OperatorTime(*it, +1));
      worm_ops[1].set_time(OperatorTime(*it, 0));
      safe_insert(ops, worm_ops[0]);
      safe_insert(ops, worm_ops[1]);

      sliding_window.compute_trace_bound(ops, trace_bound);
      std::pair<bool, EXTENDED_SCALAR> r = sliding_window.lazy_eval_trace(ops, EXTENDED_REAL(0.0), trace_bound);
      if (myabs(r.second) > trace_cutoff) {
        const SCALAR weight = convert_to_scalar(r.second / mc_config.trace);
        measure_impl(worm_ops, mc_config.sign * weight, data_);
        norm += std::abs(weight);
      }

      safe_erase(ops, worm_ops[0]);
      safe_erase(ops, worm_ops[1]);
    }

    //remove worm operators which are not shifted in time
    for (int iop = 2; iop < worm_ops.size(); ++iop) {
      safe_erase(ops, worm_ops[iop]);
    }
  }

  //normalize the data
  std::transform(data_.origin(),
                 data_.origin() + data_.num_elements(),
                 data_.origin(),
                 std::bind2nd(std::divides<std::complex<double> >(), norm));

  //pass the data to ALPS libraries
  measure_simple_vector_observable<std::complex<double> >(measurements, str.c_str(), to_std_vector(data_));

  //restore the status of the sliding window
  sliding_window.restore_state(mc_config.operators, state);
}

template<typename SCALAR>
void TwoTimeG2Measurement<SCALAR>::measure_impl(const std::vector<psi> &worm_ops, SCALAR weight,
                                                boost::multi_array<std::complex<double>, 5> &data) {
  boost::array<int, 4> flavors;
  if (worm_ops[0].time().time() != worm_ops[1].time().time()) {
    throw std::runtime_error("time worm_ops0 != time worm_ops1");
  }
  if (worm_ops[2].time().time() != worm_ops[3].time().time()) {
    throw std::runtime_error("time worm_ops2 != time worm_ops3");
  }
  double tdiff = worm_ops[0].time().time() - worm_ops[2].time().time();
  if (tdiff < 0.0) {
    tdiff += beta_;
  }
  for (int f = 0; f < 4; ++f) {
    flavors[f] = worm_ops[f].flavor();
  }

  const int num_legendre = legendre_trans_.num_legendre();
  std::vector<double> Pl_vals(num_legendre);

  legendre_trans_.compute_legendre(2 * tdiff / beta_ - 1.0, Pl_vals);
  for (int il = 0; il < num_legendre; ++il) {
    data
    [flavors[0]]
    [flavors[1]]
    [flavors[2]]
    [flavors[3]][il] += weight * legendre_trans_.get_sqrt_2l_1()[il] * Pl_vals[il];
  }
}

/**
 * @brief Generate a set of times for a given size num_times.
 * @param init_time the first element in the set
 * @param num_times the number of times to be generated (including init_time)
 * @param random alps::random
 * @param beta inverse temperature
 * @param tau_min lower bound
 * @param tau_max upper bound
 * @param operators make sure none of the entries in new_times share its position with any entry in operators.
 * @param a set of times generated
 */
inline void generate_new_time_sets(double init_time, int num_times, alps::random01 &random,
                                   double beta,
                                   double tau_min,
                                   double tau_max,
                                   const operator_container_t &operators, std::set<double> &new_times) {
  new_times.clear();
  new_times.insert(init_time);
  if (new_times.size() < num_times) {
    std::set<double> duplicate_check;
    for (operator_container_t::const_iterator it = operators.begin(); it != operators.end(); ++it) {
      duplicate_check.insert(it->time().time());
    }
    while (new_times.size() < num_times) {
      double t = open_random(random, tau_min, tau_max);
      while (t > beta) {
        t -= beta;
      }
      while (t < 0) {
        t += beta;
      }
      if (duplicate_check.find(t) == duplicate_check.end()) {
        new_times.insert(t);
        duplicate_check.insert(t);
      }
    }
    assert(new_times.size() == num_times);
  }
}

//template<int Rank>
//void init_work_space(boost::multi_array<std::complex<double>, 4 * Rank - 1> &data, int num_flavors, int num_legendre) {
//data.resize(boost::extents[num_flavors][num_flavors][num_legendre]);
//};


/*
template<typename SCALAR, int Rank>
template<typename SlidingWindow>
void GMeasurement<SCALAR, Rank>::measure(MonteCarloConfiguration<SCALAR> &mc_config,
                                         alps::accumulators::accumulator_set &measurements,
                                         alps::random01 &random,
                                         SlidingWindow &sliding_window,
                                         int average_pert_order,
                                         const std::string &str) {
  typedef typename ExtendedScalar<SCALAR>::value_type EXTENDED_SCALAR;
  typedef operator_container_t::iterator Iterator;

  //Remove the left-hand-side operators
  operator_container_t ops(mc_config.operators);
  const std::vector<psi> worm_ops_original = mc_config.p_worm->get_operators();
  safe_erase(ops, worm_ops_original);

  //Determine which operator we shift
  const int idx_operator_shifted = static_cast<int>(random() * 2 * Rank);
  const double original_time = worm_ops_original[idx_operator_shifted].time().time();

  //Generate times of the left hand operator pair c^dagger c
  const int num_times = std::max(50, average_pert_order * num_flavors_);
  double tau_min, tau_max;
  if (random() < 0.5) {
    tau_min = 0.0;
    tau_max = beta_;
  } else {
    tau_min = original_time - beta_ / average_pert_order;
    tau_max = original_time + beta_ / average_pert_order;
  }
  std::set<double> new_times;
  generate_new_time_sets(
      worm_ops_original[idx_operator_shifted].time().time(),
      num_times,
      random,
      beta_,
      tau_min,
      tau_max,
      mc_config.operators,
      new_times
  );

  //remember the current status of the sliding window
  const typename SlidingWindow::state_t state = sliding_window.get_state();
  const int n_win = sliding_window.get_n_window();

  std::vector<EXTENDED_REAL> trace_bound(sliding_window.get_num_brakets());

  //configurations whose trace is smaller than this cutoff are ignored.
  const EXTENDED_REAL trace_cutoff = EXTENDED_REAL(1.0E-30) * myabs(mc_config.trace);

  double norm = 0.0;
  std::fill(data_.origin(), data_.origin() + data_.num_elements(), 0.0);

  //insert worm operators which are not shifted in time
  std::vector<psi> worm_ops = worm_ops_original;
  for (int iop = 0; iop < worm_ops.size(); ++iop) {
    if (iop != idx_operator_shifted) {
      safe_insert(ops, worm_ops[iop]);
    }
  }

  //reset the window and move to the right most position
  sliding_window.set_window_size(
      std::max(4 * num_times, 4 * mc_config.pert_order()),
      ops, 0, ITIME_LEFT
  );

  std::vector<SCALAR> weight_flavors(num_flavors_, 0.0);

  int counter = 0;
  for (std::set<double>::iterator it = new_times.begin(); it != new_times.end(); ++it, ++counter) {
    //move the window if needed
    while (*it > sliding_window.get_tau_high()) {
      sliding_window.move_window_to_next_position(ops);
    }
    assert(*it <= sliding_window.get_tau_high());
    assert(*it >= sliding_window.get_tau_low());

    //count permutation
    const double perm_sign_change = num_operators_in_range_open(ops, *it, original_time) % 2 == 0 ? 1 : -1;

    //compute weights of samples with different flavors
    worm_ops[idx_operator_shifted].set_time(OperatorTime(*it, +1));
    std::fill(weight_flavors.begin(), weight_flavors.end(), 0.0);
    for (int flavor = 0; flavor < num_flavors_; ++flavor) {
      worm_ops[idx_operator_shifted].set_flavor(flavor);
      safe_insert(ops, worm_ops[idx_operator_shifted]);

      sliding_window.compute_trace_bound(ops, trace_bound);
      std::pair<bool, EXTENDED_SCALAR> r = sliding_window.lazy_eval_trace(ops, EXTENDED_REAL(0.0), trace_bound);
      if (myabs(r.second) > trace_cutoff) {
        weight_flavors[flavor] = perm_sign_change * convert_to_scalar(r.second / mc_config.trace);
        norm += std::abs(weight_flavors[flavor]);
      }

      safe_erase(ops, worm_ops[idx_operator_shifted]);
    }

    //measure Green's fuction for all the samples at once
    //We have to compute Legendre coefficients and exponents only once.
    measure_impl(worm_ops, idx_operator_shifted, mc_config.sign, weight_flavors);
  }

  //normalize the data
  std::transform(data_.origin(),
                 data_.origin() + data_.num_elements(),
                 data_.origin(),
                 std::bind2nd(std::divides<std::complex<double> >(), norm));

  //pass the data to ALPS libraries
  measure_simple_vector_observable<std::complex<double> >(measurements, str.c_str(), to_std_vector(data_));

  //restore the status of the sliding window
  sliding_window.restore_state(mc_config.operators, state);
}
*/

inline int compute_perm_sign(const std::vector<psi> &ops) {
  std::vector<OperatorTime> times;
  times.reserve(ops.size());
  for (int i = 0; i < ops.size(); ++i) {
    times.push_back(ops[i].time());
  }
  assert(times.size() == ops.size());
  return alps::fastupdate::comb_sort(times.begin(), times.end(), std::greater<OperatorTime>());
}

template<typename SCALAR, int Rank>
void GMeasurement<SCALAR, Rank>::measure_via_hyb(const MonteCarloConfiguration<SCALAR> &mc_config,
                                                 alps::accumulators::accumulator_set &measurements,
                                                 alps::random01 &random,
                                                 const std::string &str,
                                                 double eps) {
  typedef typename ExtendedScalar<SCALAR>::value_type EXTENDED_SCALAR;
  typedef operator_container_t::iterator Iterator;
  typedef Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

  const int pert_order = mc_config.pert_order();
  boost::shared_ptr<HybridizationFunction<SCALAR> > p_gf = mc_config.M.get_greens_function();
  const std::vector<psi> cdagg_ops = mc_config.M.get_cdagg_ops();
  const std::vector<psi> c_ops = mc_config.M.get_c_ops();
  const std::vector<psi> worm_ops = mc_config.p_worm->get_operators();

  //compute the intermediate state by connecting operators in the worm by hybridization
  alps::fastupdate::ResizableMatrix<SCALAR> M(pert_order + Rank, pert_order + Rank, 0.0);
  M.conservative_resize(pert_order, pert_order);
  int offset = 0;
  for (int ib = 0; ib < mc_config.M.num_blocks(); ++ib) {
    const int block_size = mc_config.M.block_matrix_size(ib);
    M.block(offset, offset, block_size, block_size) = mc_config.M.compute_inverse_matrix(ib);
    offset += block_size;
  }
  matrix_t B(pert_order, Rank), C(Rank, pert_order), D(Rank, Rank);
  for (int i = 0; i < pert_order; ++i) {
    for (int j = 0; j < Rank; ++j) {
      B(i, j) = p_gf->operator()(c_ops[i], worm_ops[2 * j + 1]);
    }
  }
  for (int i = 0; i < Rank; ++i) {
    for (int j = 0; j < pert_order; ++j) {
      C(i, j) = p_gf->operator()(worm_ops[2 * i], cdagg_ops[j]);
    }
  }
  for (int i = 0; i < Rank; ++i) {
    for (int j = 0; j < Rank; ++j) {
      D(i, j) = p_gf->operator()(worm_ops[2 * i], worm_ops[2 * j + 1]);
    }
  }
  //check quantum number conservation
  std::valarray<int> block_size_diff(0, mc_config.M.num_blocks());
  for (int i = 0; i < Rank; ++i) {
    ++block_size_diff[mc_config.M.block_belonging_to(worm_ops[2 * i].flavor())];//annihilation operator
    --block_size_diff[mc_config.M.block_belonging_to(worm_ops[2 * i + 1].flavor())];//creation operator
  }

  //add aux field to avoid a singular matrix in the case quantum number is not conserved for the intermediate state
  if (block_size_diff.max() != 0 || block_size_diff.min() != 0) {
    for (int i = 0; i < Rank; ++i) {
      D(i, i) += random() < 0.5 ? eps : -eps;
    }
  }

  const SCALAR det_rat = alps::fastupdate::compute_det_ratio_up(B, C, D, M);
  alps::fastupdate::compute_inverse_matrix_up(B, C, D, M);
  assert(det_rat != 0.0);
  assert(M.size1() == pert_order + 1);

  /*
   * At this point, it should be noted that the weight of a Monte Carlo configuration is given by
   *   w = determinant * trace * permutation_sign,
   *  where the permutation_sign is from time ordering of operators in the trace.
   *  And, the determinant is computed assumed that its row and cols are time-ordered.
   *  But, there are not always time ordered in memory for performance reasons.
   *  So, the weight may be recast into the form
   *   w = determinant_actual_ordering * permutation_sign_row_col * trace * permutation_sign.
   *
   *  Here, we perform a brute-force evaluation of permutation_sign_row_col and permutation_sign.
   */
  std::vector<psi> cdagg_ops_new(cdagg_ops);
  std::vector<psi> c_ops_new(c_ops);
  for (int i = 0; i < Rank; ++i) {
    c_ops_new.push_back(worm_ops[2 * i]);
    cdagg_ops_new.push_back(worm_ops[2 * i + 1]);
  }
  const int perm_sign_trace_rat =
      compute_permutation_sign_impl(cdagg_ops_new, c_ops_new, std::vector<psi>())
          / compute_permutation_sign_impl(cdagg_ops, c_ops, worm_ops);

  const int perm_sign_det_rat =
      (std::count_if(cdagg_ops.begin(), cdagg_ops.end(), std::bind2nd(std::less<psi>(), worm_ops[1])) +
          std::count_if(c_ops.begin(), c_ops.end(), std::bind2nd(std::less<psi>(), worm_ops[0]))) % 2 == 0 ? 1 : -1;

  //weight_intermediate_state/weigh_current_state
  //We'are ready for measuring Green's function using M and weight_rat here.
  //Note: sign of the intermediate state is sign(weight_rat * sign)
  const SCALAR weight_rat = (1. * perm_sign_det_rat * perm_sign_trace_rat) * det_rat;

  //drop small values
  const double cutoff = 1.0e-15 * M.block().cwiseAbs().maxCoeff();
  for (int j = 0; j < M.size2(); ++j) {
    for (int i = 0; i < M.size1(); ++i) {
      if (std::abs(M(i, j)) < cutoff) {
        M(i, j) = 0.0;
      }
    }
  }

  //measure by removal as we would do for the partition function expansion
  MeasureGHelper<SCALAR, Rank>::perform(beta_,
                                        legendre_trans_,
                                        num_freq_,
                                        mc_config.sign,
                                        weight_rat,
                                        cdagg_ops_new,
                                        c_ops_new,
                                        M,
                                        data_);

  //pass the data to ALPS libraries
  measure_simple_vector_observable<std::complex<double> >(measurements, str.c_str(), to_std_vector(data_));
}

//Measure G1 by removal in G1 space
template<typename SCALAR>
void MeasureGHelper<SCALAR, 1>::perform(double beta,
                                        LegendreTransformer &legendre_trans,
                                        int n_freq,
                                        SCALAR sign,
                                        SCALAR weight_rat_intermediate_state,
                                        const std::vector<psi> &creation_ops,
                                        const std::vector<psi> &annihilation_ops,
                                        const alps::fastupdate::ResizableMatrix<SCALAR> &M,
                                        boost::multi_array<std::complex<double>, 3> &result) {
  const double temperature = 1. / beta;
  const int num_flavors = result.shape()[0];
  const int num_legendre = legendre_trans.num_legendre();

  std::vector<double> Pl_vals(num_legendre);
  std::fill(result.origin(), result.origin() + result.num_elements(), 0.0);

  double norm = 0.0;
  std::vector<psi>::const_iterator it1, it2;
  for (int k = 0; k < M.size1(); k++) {
    (k == 0 ? it1 = annihilation_ops.begin() : it1++);
    for (int l = 0; l < M.size2(); l++) {
      (l == 0 ? it2 = creation_ops.begin() : it2++);
      if (M(l, k) == 0.0) {
        continue;
      }
      double argument = it1->time() - it2->time();
      double bubble_sign = 1;
      if (argument > 0) {
        bubble_sign = 1;
      } else {
        bubble_sign = -1;
        argument += beta;
      }
      assert(-0.01 < argument && argument < beta + 0.01);

      const int flavor_a = it1->flavor();
      const int flavor_c = it2->flavor();
      const double x = 2 * argument * temperature - 1.0;
      legendre_trans.compute_legendre(x, Pl_vals);
      norm += std::abs(weight_rat_intermediate_state * M(l, k));
      const SCALAR coeff = M(l, k) * bubble_sign * sign * weight_rat_intermediate_state;
      for (int il = 0; il < num_legendre; ++il) {
        result[flavor_a][flavor_c][il] += coeff * legendre_trans.get_sqrt_2l_1()[il] * Pl_vals[il];
      }
    }
  }

  //normalization
  std::transform(result.origin(), result.origin() + result.num_elements(), result.origin(),
                 std::bind2nd(std::divides<std::complex<double> >(), -1.0 * norm * beta));
};

//Measure G2 by removal in G2 space
template<typename SCALAR>
void MeasureGHelper<SCALAR, 2>::perform(double beta,
                                        LegendreTransformer &legendre_trans,
                                        int n_freq,
                                        SCALAR sign,
                                        SCALAR weight_rat_intermediate_state,
                                        const std::vector<psi> &creation_ops,
                                        const std::vector<psi> &annihilation_ops,
                                        const alps::fastupdate::ResizableMatrix<SCALAR> &M,
                                        boost::multi_array<std::complex<double>, 7> &result) {
  const double temperature = 1. / beta;
  const int num_flavors = result.shape()[0];
  const int num_legendre = legendre_trans.num_legendre();
  const int pert_order = creation_ops.size();

  //Compute values of P
  std::vector<double> sqrt_2l_1 = legendre_trans.get_sqrt_2l_1();
  std::vector<double> sqrt_2l_1_p(sqrt_2l_1);
  for (int il = 0; il < num_legendre; il += 2) {
    sqrt_2l_1_p[il] *= -1;
  }

  boost::multi_array<double, 3>
      Pl(boost::extents[pert_order][pert_order][num_legendre]);//annihilator, creator, legendre
  {
    std::vector<double> Pl_tmp(num_legendre);
    for (int k = 0; k < M.size1(); k++) {
      for (int l = 0; l < M.size2(); l++) {
        double argument = annihilation_ops[k].time() - creation_ops[l].time();
        double arg_sign = 1.0;
        if (argument < 0) {
          argument += beta;
          arg_sign = -1.0;
        }
        const double x = 2 * argument * temperature - 1.0;
        legendre_trans.compute_legendre(x, Pl_tmp);
        for (int il = 0; il < num_legendre; ++il) {
          Pl[k][l][il] = arg_sign * Pl_tmp[il];
        }
      }
    }
  }

  boost::multi_array<std::complex<double>, 3>
      expiomega(boost::extents[pert_order][pert_order][n_freq]);//annihilator, creator, legendre
  {
    for (int k = 0; k < M.size1(); k++) {
      for (int l = 0; l < M.size2(); l++) {
        const double tau_diff = annihilation_ops[k].time() - creation_ops[l].time();
        expiomega[k][l][0] = std::exp(std::complex<double>(0.0, 2 * M_PI * tau_diff * temperature));
        std::complex<double> rat = expiomega[k][l][0];
        for (int freq = 1; freq < n_freq; ++freq) {
          expiomega[k][l][freq] = rat * expiomega[k][l][freq - 1];
        }
      }
    }
  }

  //naive way to evaluate
  //The indices of M are reverted from (C. 24) of L. Boehnke (2011) because we're using the F convention here.
  std::fill(result.origin(), result.origin() + result.num_elements(), 0.0);
  double norm = 0.0;
  for (int a = 0; a < pert_order; ++a) {
    const int flavor_a = annihilation_ops[a].flavor();
    for (int b = 0; b < pert_order; ++b) {
      const int flavor_b = creation_ops[b].flavor();
      for (int c = 0; c < pert_order; ++c) {
        const int flavor_c = annihilation_ops[c].flavor();
        for (int d = 0; d < pert_order; ++d) {
          const int flavor_d = creation_ops[d].flavor();

          const SCALAR coeff = sign * weight_rat_intermediate_state * (M(b, a) * M(d, c) - M(d, a) * M(c, b));
          norm += std::abs(coeff);
          for (int il = 0; il < num_legendre; ++il) {
            for (int il_p = 0; il_p < num_legendre; ++il_p) {
              for (int im = 0; im < n_freq; ++im) {
                result[flavor_a][flavor_b][flavor_c][flavor_d][il][il_p][im] +=
                    coeff * sqrt_2l_1[il] * sqrt_2l_1_p[il_p] * Pl[a][b][il] * Pl[c][d][il_p] * expiomega[a][d][im];
              }
            }
          }
        }
      }
    }
  }

  //normalization
  std::transform(result.origin(), result.origin() + result.num_elements(), result.origin(),
                 std::bind2nd(std::divides<std::complex<double> >(), norm * beta));
};

//Measure single-particle Green's function using Legendre polynomials
/*
template<typename SCALAR, int Rank>
typename boost::enable_if_c<Rank == 1, SCALAR>::type
GMeasurement<SCALAR, Rank>::measure_impl(const std::vector<psi> &worm_ops,
                                         int idx_operator_shifted,
                                         SCALAR sign,
                                         std::vector<SCALAR> &weight_flavors) {
  double tdiff = worm_ops[0].time().time() - worm_ops[1].time().time();
  double coeff = 1.0;
  if (tdiff < 0.0) {
    tdiff += beta_;
    coeff *= -1;
  }

  const int num_legendre = legendre_trans_.num_legendre();
  std::vector<double> Pl_vals(num_legendre);
  legendre_trans_.compute_legendre(2 * tdiff / beta_ - 1.0, Pl_vals);

  boost::array<int, 2> flavors;
  for (int iop = 0; iop < 2; ++iop) {
    flavors[iop] = worm_ops[iop].flavor();
  }
  for (int flavor = 0; flavor < num_flavors_; ++flavor) {
    flavors[idx_operator_shifted] = flavor;
    const SCALAR coeff2 = sign * weight_flavors[flavor] * coeff;
    for (int il = 0; il < num_legendre; ++il) {
      data_[flavors[0]][flavors[1]][il] += coeff2 * legendre_trans_.get_sqrt_2l_1()[il] * Pl_vals[il];
    }
  }

  return 0.0; //dummy
}
*/

template<typename SCALAR, int Rank>
void
EqualTimeGMeasurement<SCALAR, Rank>::measure_G1(MonteCarloConfiguration<SCALAR> &mc_config,
                                                alps::accumulators::accumulator_set &measurements,
                                                const std::string &str) {
  boost::multi_array<std::complex<double>, 2> data;
  data.resize(boost::extents[num_flavors_][num_flavors_]);
  std::fill(data.origin(), data.origin() + data.num_elements(), 0.0);

  data[mc_config.p_worm->get_flavor(0)][mc_config.p_worm->get_flavor(1)] = mc_config.sign;

  measure_simple_vector_observable<std::complex<double> >(measurements, str.c_str(), to_std_vector(data));
};

template<typename SCALAR, int Rank>
void
EqualTimeGMeasurement<SCALAR, Rank>::measure_G2(MonteCarloConfiguration<SCALAR> &mc_config,
                                                alps::accumulators::accumulator_set &measurements,
                                                const std::string &str) {
  boost::multi_array<std::complex<double>, 4> data;
  data.resize(boost::extents[num_flavors_][num_flavors_][num_flavors_][num_flavors_]);
  std::fill(data.origin(), data.origin() + data.num_elements(), 0.0);

  data[mc_config.p_worm->get_flavor(0)][mc_config.p_worm->get_flavor(1)]
  [mc_config.p_worm->get_flavor(2)][mc_config.p_worm->get_flavor(3)] = mc_config.sign;

  measure_simple_vector_observable<std::complex<double> >(measurements, str.c_str(), to_std_vector(data));
};
