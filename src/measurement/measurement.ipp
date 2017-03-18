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

template<typename SCALAR, int Rank>
void GMeasurement<SCALAR, Rank>::measure_via_hyb(const MonteCarloConfiguration<SCALAR> &mc_config,
                                                 alps::accumulators::accumulator_set &measurements,
                                                 alps::random01 &random,
                                                 int max_num_ops,
                                                 double eps) {
  typedef typename ExtendedScalar<SCALAR>::value_type EXTENDED_SCALAR;
  typedef operator_container_t::iterator Iterator;
  typedef Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

  const int pert_order = mc_config.pert_order();
  boost::shared_ptr<HybridizationFunction<SCALAR> > p_gf = mc_config.M.get_greens_function();
  const std::vector<psi> cdagg_ops = mc_config.M.get_cdagg_ops();
  const std::vector<psi> c_ops = mc_config.M.get_c_ops();
  const std::vector<psi> worm_ops = mc_config.p_worm->get_operators();

  const int n_aux_lines = Rank;

  //compute the intermediate state by connecting operators in the worm by hybridization
  alps::fastupdate::ResizableMatrix<SCALAR> M(pert_order + Rank + n_aux_lines, pert_order + Rank + n_aux_lines, 0.0);
  M.conservative_resize(pert_order, pert_order);
  int offset = 0;
  for (int ib = 0; ib < mc_config.M.num_blocks(); ++ib) {
    const int block_size = mc_config.M.block_matrix_size(ib);
    M.block(offset, offset, block_size, block_size) = mc_config.M.compute_inverse_matrix(ib);
    offset += block_size;
  }
  matrix_t B(pert_order, Rank + n_aux_lines), C(Rank + n_aux_lines, pert_order), D(Rank + n_aux_lines, Rank + n_aux_lines);
  B.setZero();
  C.setZero();
  D.setZero();
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
  for (int i = 0; i < Rank + n_aux_lines; ++i) {
    for (int j = 0; j < Rank + n_aux_lines; ++j) {
      if (i < Rank && j < Rank) {
        D(i, j) = p_gf->operator()(worm_ops[2 * i], worm_ops[2 * j + 1]);
      } else {
        //avoid a singular matrix
        D(i, j) = eps * random();
      }
    }
  }

  const SCALAR det_rat = alps::fastupdate::compute_det_ratio_up(B, C, D, M);
  if (det_rat == 0.0) {
    std::cerr << "Warning intermediate state has a vanishing weight in measurement of G" << Rank << "!" << std::endl;
    return;
  }
  alps::fastupdate::compute_inverse_matrix_up(B, C, D, M);
  assert(M.size1() == pert_order + Rank + n_aux_lines);

  std::vector<psi> cdagg_ops_new(cdagg_ops);
  std::vector<psi> c_ops_new(c_ops);
  for (int i = 0; i < Rank; ++i) {
    c_ops_new.push_back(worm_ops[2 * i]);
    cdagg_ops_new.push_back(worm_ops[2 * i + 1]);
  }
  const SCALAR weight_rat = det_rat;

  //TO DO: move this to a separated function
  if (pert_order + Rank > max_num_ops) {
    const int num_ops = pert_order + Rank;
    std::vector<bool> is_row_active(num_ops + n_aux_lines, false), is_col_active(num_ops + n_aux_lines, false);
    //always choose the original worm position for detailed balance condition
    for (int i = 0; i < Rank + n_aux_lines; ++i) {
      //is_row_active[num_ops - i] = true;
      //is_col_active[num_ops - i] = true;
      is_row_active[is_row_active.size() - 1 - i] = true;
      is_col_active[is_col_active.size() - 1 - i] = true;
    }
    for (int i = 0; i < max_num_ops - Rank; ++i) {
      is_row_active[i] = true;
      is_col_active[i] = true;
    }
    MyRandomNumberGenerator rnd(random);
    std::random_shuffle(is_row_active.begin(), is_row_active.begin() + pert_order, rnd);
    std::random_shuffle(is_col_active.begin(), is_col_active.begin() + pert_order, rnd);
    assert(boost::count(is_col_active, true) == max_num_ops + n_aux_lines);
    assert(boost::count(is_row_active, true) == max_num_ops + n_aux_lines);

    {
      std::vector<psi> cdagg_ops_reduced, c_ops_reduced;
      for (int i = 0; i < num_ops; ++i) {
        if (is_col_active[i]) {
          c_ops_reduced.push_back(c_ops_new[i]);
        }
        if (is_row_active[i]) {
          cdagg_ops_reduced.push_back(cdagg_ops_new[i]);
        }
      }
      std::swap(cdagg_ops_reduced, cdagg_ops_new);
      std::swap(c_ops_reduced, c_ops_new);
      assert(cdagg_ops_new.size() == max_num_ops);
      assert(c_ops_new.size() == max_num_ops);
    }

    {
      const int mat_size = M.size1();
      alps::fastupdate::ResizableMatrix<SCALAR> M_reduced(max_num_ops + n_aux_lines, max_num_ops + n_aux_lines, 0.0);
      int j_reduced = 0;
      for (int j = 0; j < mat_size; ++j) {
        if (!is_col_active[j]) {
          continue;
        }
        int i_reduced = 0;
        for (int i = 0; i < mat_size; ++i) {
          if (!is_row_active[i]) {
            continue;
          }
          M_reduced(i_reduced, j_reduced) = M(i, j);
          ++ i_reduced;
        }
        ++ j_reduced;
        assert(i_reduced == max_num_ops + n_aux_lines);
      }
      assert(j_reduced == max_num_ops + n_aux_lines);
      std::swap(M, M_reduced);
      assert(M.size1() == max_num_ops + n_aux_lines);
      assert(M.size2() == max_num_ops + n_aux_lines);
    }
  }

  //drop small values
  const double cutoff = 1.0e-10 * M.block().cwiseAbs().maxCoeff();
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
  ++ num_data_;

  if (num_data_ == max_num_data_) {
    //pass the data to ALPS libraries
    std::transform(data_.origin(), data_.origin() + data_.num_elements(), data_.origin(),
                   std::bind2nd(std::divides<std::complex<double> >(), 1. * max_num_data_));
    measure_simple_vector_observable<std::complex<double> >(measurements, str_.c_str(), to_std_vector(data_));

    num_data_ = 0;
    std::fill(data_.origin(), data_.origin() + data_.num_elements(), 0.0);
  }
}

//template<typename SCALAR>
//struct RescaleAdd {
  //RescaleAdd(SCALAR a) : a_(a) {}
  //SCALAR operator()(const SCALAR &x, const SCALAR &y) {
    //return a_ * x + y;
  //}
  //SCALAR a_;
//};

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

  std::vector<psi>::const_iterator it1, it2;
  const int mat_size = M.size1();

  //First, we compute relative weights
  boost::multi_array<SCALAR,2> coeffs(boost::extents[mat_size-1][mat_size-1]);
  double norm = 0.0;
  for (int k = 0; k < mat_size - 1; k++) {//the last one is aux fields
    (k == 0 ? it1 = annihilation_ops.begin() : it1++);
    for (int l = 0; l < mat_size - 1; l++) {
      (l == 0 ? it2 = creation_ops.begin() : it2++);
      if (M(l, k) == 0.0) {
        coeffs[k][l] = 0.0;
        continue;
      }

      const double bubble_sign = it1->time() - it2->time() > 0.0 ? 1.0 : -1.0;

      coeffs[k][l] = (M(l, k) * M(mat_size - 1, mat_size - 1) - M(l, mat_size - 1) * M(mat_size - 1, k))
          * bubble_sign * sign * weight_rat_intermediate_state;
      norm += std::abs(coeffs[k][l]);
    }
  }

  double scale_fact = -1.0/(norm * beta);
  for (int k = 0; k < mat_size - 1; k++) {//the last one is aux fields
    (k == 0 ? it1 = annihilation_ops.begin() : it1++);
    for (int l = 0; l < mat_size - 1; l++) {
      (l == 0 ? it2 = creation_ops.begin() : it2++);
      if (M(l, k) == 0.0) {
        continue;
      }
      double argument = it1->time() - it2->time();
      if (argument < 0) {
        argument += beta;
      }
      assert(-0.01 < argument && argument < beta + 0.01);

      const int flavor_a = it1->flavor();
      const int flavor_c = it2->flavor();
      const double x = 2 * argument * temperature - 1.0;
      legendre_trans.compute_legendre(x, Pl_vals);
      for (int il = 0; il < num_legendre; ++il) {
        result[flavor_a][flavor_c][il] += scale_fact * coeffs[k][l] * legendre_trans.get_sqrt_2l_1()[il] * Pl_vals[il];
      }
    }
  }
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
  const int num_phys_rows = creation_ops.size();
  //std::cout << "num_phys_rows " << num_phys_rows << " " << annihilation_ops.size() << " " << M.size1() << std::endl;
  const int n_aux_lines = 2;
  if (creation_ops.size() != annihilation_ops.size() || creation_ops.size() != M.size1() - n_aux_lines) {
    throw std::runtime_error("Fatal error in MeasureGHelper<SCALAR, 2>::perform()");
  }

  //Compute values of P
  std::vector<double> sqrt_2l_1 = legendre_trans.get_sqrt_2l_1();
  std::vector<double> sqrt_2l_1_p(sqrt_2l_1);
  for (int il = 0; il < num_legendre; il += 2) {
    sqrt_2l_1_p[il] *= -1;
  }

  boost::multi_array<double, 3>
      sqrt_2l_1_Pl(boost::extents[num_phys_rows][num_phys_rows][num_legendre]);//annihilator, creator, legendre
  boost::multi_array<double, 3>
      sqrt_2l_1_Pl_p(boost::extents[num_phys_rows][num_phys_rows][num_legendre]);//annihilator, creator, legendre
  {
    std::vector<double> Pl_tmp(num_legendre);
    for (int k = 0; k < num_phys_rows; k++) {
      for (int l = 0; l < num_phys_rows; l++) {
        double argument = annihilation_ops[k].time() - creation_ops[l].time();
        double arg_sign = 1.0;
        if (argument < 0) {
          argument += beta;
          arg_sign = -1.0;
        }
        const double x = 2 * argument * temperature - 1.0;
        legendre_trans.compute_legendre(x, Pl_tmp);
        for (int il = 0; il < num_legendre; ++il) {
          sqrt_2l_1_Pl[k][l][il] = arg_sign * Pl_tmp[il] * sqrt_2l_1[il];
          sqrt_2l_1_Pl_p[k][l][il] = arg_sign * Pl_tmp[il] * sqrt_2l_1_p[il];
        }
      }
    }
  }

  boost::multi_array<std::complex<double>, 3>
      expiomega(boost::extents[num_phys_rows][num_phys_rows][n_freq]);//annihilator, creator, legendre
  {
    for (int k = 0; k < num_phys_rows; k++) {
      for (int l = 0; l < num_phys_rows; l++) {
        const double tau_diff = annihilation_ops[k].time() - creation_ops[l].time();
        const std::complex<double> rat = std::exp(std::complex<double>(0.0, 2 * M_PI * tau_diff * temperature));
        expiomega[k][l][0] = 1.0;
        for (int freq = 1; freq < n_freq; ++freq) {
          expiomega[k][l][freq] = rat * expiomega[k][l][freq - 1];
        }
      }
    }
  }

  //naive way to evaluate
  //The indices of M are reverted from (C. 24) of L. Boehnke (2011) because we're using the F convention here.

  //First, compute relative weights
  const int rank = 2;
  const int det_size = rank + n_aux_lines;
  Eigen::Matrix<SCALAR,det_size,det_size> tmp_mat;
  boost::array<int,det_size> rows3, cols3;
  const int last = M.size1() - 1;
  for (int i = 0; i < n_aux_lines; ++i) {
    cols3[rank+i] = rows3[rank+i] = i + M.size1() - n_aux_lines;
  }
  assert(cols3.back()==last);
  boost::multi_array<SCALAR,4> coeffs(boost::extents[num_phys_rows][num_phys_rows][num_phys_rows][num_phys_rows]);
  double norm = 0.0;
  for (int a = 0; a < num_phys_rows; ++a) {
    for (int b = 0; b < num_phys_rows; ++b) {
      for (int c = 0; c < num_phys_rows; ++c) {
        if (a==c) {
          continue;
        }
        for (int d = 0; d < num_phys_rows; ++d) {
          if (b==d) {
            continue;
          }
          /*
           * Delta convention
           * M_ab  M_ad  M_a*
           * M_cb  M_cd  M_c*
           * M_*b  M_*d  M_**
           */
          rows3[0] = b;
          rows3[1] = d;
          cols3[0] = a;
          cols3[1] = c;
          for (int j = 0; j < det_size; ++j) {
            for (int i = 0; i < det_size; ++i) {
              tmp_mat(i,j) = M(rows3[i], cols3[j]);
            }
          }
          coeffs[a][b][c][d] = sign * weight_rat_intermediate_state * tmp_mat.determinant();
          norm += std::abs(coeffs[a][b][c][d]);
        }
      }
    }
  }

  //Then, accumulate data
  const double scale_fact = 1.0/(norm * beta);
  for (int a = 0; a < num_phys_rows; ++a) {
    const int flavor_a = annihilation_ops[a].flavor();
    for (int b = 0; b < num_phys_rows; ++b) {
      const int flavor_b = creation_ops[b].flavor();
      for (int c = 0; c < num_phys_rows; ++c) {
        if (a==c) {
          continue;
        }
        const int flavor_c = annihilation_ops[c].flavor();
        for (int d = 0; d < num_phys_rows; ++d) {
          if (b==d) {
            continue;
          }
          const int flavor_d = creation_ops[d].flavor();

          if (coeffs[a][b][c][d] == 0.0) {
            continue;
          }
          const SCALAR coeff = coeffs[a][b][c][d] * scale_fact;
          for (int il = 0; il < num_legendre; ++il) {
            for (int il_p = 0; il_p < num_legendre; ++il_p) {
              const SCALAR coeff2 = coeff * sqrt_2l_1_Pl[a][b][il] * sqrt_2l_1_Pl_p[c][d][il_p];
              for (int im = 0; im < n_freq; ++im) {
                result[flavor_a][flavor_b][flavor_c][flavor_d][il][il_p][im] += coeff2 * expiomega[a][d][im];
              }
            }
          }
        }
      }
    }
  }
};

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
