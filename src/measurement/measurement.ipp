#include "./measurement.hpp"

/*
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
*/

std::pair<double,double>
mod_beta(double tau, double beta) {
  if (tau > 0 && tau < beta) {
      return std::make_pair(tau, 1.0);
  } else if (tau < 0 && tau > -beta) {
    return std::make_pair(tau + beta, -1.0);
  } else {
    throw std::runtime_error("error in mod_beta!");
  }
};

template<typename SCALAR>
Reconnections<SCALAR>::Reconnections(const MonteCarloConfiguration<SCALAR> &mc_config,
                                                 alps::random01 &random,
                                                 int max_num_ops,
                                                 int Rank,
                                                 double eps) {
  typedef operator_container_t::iterator Iterator;
  typedef Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

  const int pert_order = mc_config.pert_order();
  boost::shared_ptr<HybridizationFunction<SCALAR> > p_gf = mc_config.M.get_greens_function();
  const std::vector<psi> cdagg_ops = mc_config.M.get_cdagg_ops();
  const std::vector<psi> c_ops = mc_config.M.get_c_ops();
  const std::vector<psi> worm_ops = mc_config.p_worm->get_operators();

  const int n_aux_lines = Rank;

  //compute the intermediate state by connecting operators in the worm by hybridization
  M_.destructive_resize(pert_order + Rank + n_aux_lines, pert_order + Rank + n_aux_lines);

  M_.conservative_resize(pert_order, pert_order);
  for (int i=0; i<pert_order; ++i) {
    for (int j=0; j<pert_order; ++j) {
        M_(i, j) = 0;
    }
  }
  int offset = 0;
  for (int ib = 0; ib < mc_config.M.num_blocks(); ++ib) {
    const int block_size = mc_config.M.block_matrix_size(ib);
    M_.block(offset, offset, block_size, block_size) = mc_config.M.compute_inverse_matrix(ib);
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

  const SCALAR det_rat = alps::fastupdate::compute_det_ratio_up(B, C, D, M_);
  if (det_rat == 0.0) {
    std::cerr << "Warning intermediate state has a vanishing weight in measurement of G" << Rank << "!" << std::endl;
    return;
  }
  alps::fastupdate::compute_inverse_matrix_up(B, C, D, M_);
  assert(M_.size1() == pert_order + Rank + n_aux_lines);

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
      const int mat_size = M_.size1();
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
          M_reduced(i_reduced, j_reduced) = M_(i, j);
          ++ i_reduced;
        }
        ++ j_reduced;
        assert(i_reduced == max_num_ops + n_aux_lines);
      }
      assert(j_reduced == max_num_ops + n_aux_lines);
      std::swap(M_, M_reduced);
      assert(M_.size1() == max_num_ops + n_aux_lines);
      assert(M_.size2() == max_num_ops + n_aux_lines);
    }
  }

  //drop small values
  const double cutoff = 1.0e-10 * M_.block().cwiseAbs().maxCoeff();
  for (int j = 0; j < M_.size2(); ++j) {
    for (int i = 0; i < M_.size1(); ++i) {
      if (std::abs(M_(i, j)) < cutoff) {
        M_(i, j) = 0.0;
      }
    }
  }

  creation_ops_ = cdagg_ops_new;
  annihilation_ops_ = c_ops_new;
  weight_rat_intermediate_state_ = weight_rat;
}


//Compute G1 by removal in G1 space
template<typename SCALAR>
void compute_G1(const IRbasis &basis,
                const MonteCarloConfiguration<SCALAR>& mc_config,
                const Reconnections<SCALAR>& reconnect,
                boost::multi_array<std::complex<double>, 3> &result,
                boost::multi_array<std::complex<double>, 3> &bin_result
) {
  double beta = basis.beta();

  std::vector<double> Ultau_vals(static_cast<int>(basis.dim_F()));
  const auto& M = reconnect.M();
  const auto& creation_ops = reconnect.creation_ops();
  const auto& annihilation_ops = reconnect.annihilation_ops();
  SCALAR sign = mc_config.sign;

  std::vector<psi>::const_iterator it1, it2;
  const int mat_size = M.size1();

  //First, we compute relative weights
  boost::multi_array<SCALAR,2> wprimes(boost::extents[mat_size-1][mat_size-1]);
  double sum_wprime = 0.0;
  for (int k = 0; k < mat_size - 1; k++) {//the last one is aux fields
    (k == 0 ? it1 = annihilation_ops.begin() : it1++);
    for (int l = 0; l < mat_size - 1; l++) {
      (l == 0 ? it2 = creation_ops.begin() : it2++);
      if (M(l, k) == 0.0) {
        wprimes[k][l] = 0.0;
        continue;
      }

      double bubble_sign = it1->time() - it2->time() > 0.0 ? 1.0 : -1.0;

      wprimes[k][l] = (M(l, k) * M(mat_size - 1, mat_size - 1) - M(l, mat_size - 1) * M(mat_size - 1, k))
          * bubble_sign * sign * reconnect.weight_rat_intermediate_state();
      wprimes[k][l] *= GWorm<1>::get_weight_correction(it1->time().time(), it2->time().time(), mc_config.p_irbasis);

      sum_wprime += std::abs(wprimes[k][l]);
    }
  }

  // Ingredients of scale_fact.
  // beta is due to the degree of freedom of origin of the relative times.
  // The "minus" is from the definition of the Green's function. G(tau) = - <T c^dagger(tau) c(0)>.
  double scale_fact = -1.0/(sum_wprime * beta);
  for (int k = 0; k < mat_size - 1; k++) {//the last one is aux fields
    (k == 0 ? it1 = annihilation_ops.begin() : it1++);
    for (int l = 0; l < mat_size - 1; l++) {
      (l == 0 ? it2 = creation_ops.begin() : it2++);
      if (M(l, k) == 0.0) {
        continue;
      }
      double argument = it1->time() - it2->time();
      double rw_corr = GWorm<1>::get_weight_correction(it1->time().time(), it2->time().time(), mc_config.p_irbasis);
      if (argument < 0) {
        argument += beta;
        // Note: sign change is already included in the definition of coeffs as "bubble sign".
      }
      assert(-0.01 < argument && argument < beta + 0.01);

      const int flavor_a = it1->flavor();
      const int flavor_c = it2->flavor();
      basis.compute_Utau_F(argument, Ultau_vals);
      SCALAR coeff = wprimes[k][l] * scale_fact / rw_corr;
      for (int il = 0; il < basis.dim_F(); ++il) {
        result[flavor_a][flavor_c][il] += Ultau_vals[il] * coeff;
      }

      auto idx = mc_config.p_irbasis->get_bin_index(argument);
      auto bin_length = mc_config.p_irbasis->bin_edges()[idx+1] - mc_config.p_irbasis->bin_edges()[idx];
      bin_result[flavor_a][flavor_c][idx] += coeff/bin_length;
    }
  }
}

//Compute G2 by removal in G2 space
template<typename SCALAR>
void compute_G2(const IRbasis &basis,
                int num_freq_f,
                int num_freq_b,
                const MonteCarloConfiguration<SCALAR>& mc_config,
                const Reconnections<SCALAR>& reconnect,
                boost::multi_array<std::complex<double>, 7> &result
) {
  double beta = basis.beta();

  const auto &M = reconnect.M();
  const auto &creation_ops = reconnect.creation_ops();
  const auto &annihilation_ops = reconnect.annihilation_ops();
  SCALAR sign = mc_config.sign;

  const int mat_size = M.size1();

  if (num_freq_f%2 == 1) {
    throw std::runtime_error("num_freq_f must be even!");
  }

  //First, we compute relative weights
  const double temperature = 1. / beta;
  const int num_flavors = result.shape()[0];
  const int num_phys_rows = creation_ops.size();
  const int n_aux_lines = 2;
  if (creation_ops.size() != annihilation_ops.size() || creation_ops.size() != M.size1() - n_aux_lines) {
    throw std::runtime_error("Fatal error in compute_G2");
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

  // List of fermion frequencies
  auto iwn_f_min = - num_freq_f/2;
  double min_freq_f = M_PI * (2*iwn_f_min + 1)/beta;

  boost::multi_array<std::complex<double>, 3>
          exp_f(boost::extents[num_phys_rows][num_phys_rows][num_freq_f]);//annihilator, creator, freq_f
  boost::multi_array<std::complex<double>, 3>
          exp_b(boost::extents[num_phys_rows][num_phys_rows][num_freq_b]);//annihilator, creator, freq_b
  double tau_diff, sign_mod;
  for (int k = 0; k < num_phys_rows; k++) {
    for (int l = 0; l < num_phys_rows; l++) {
      double argument = annihilation_ops[k].time() - creation_ops[l].time();

      std::tie(tau_diff, sign_mod) = mod_beta(argument, beta);

      auto rat = std::exp(std::complex<double>(0.0, 2 * M_PI * tau_diff / beta));

      exp_f[k][l][0] = std::exp(std::complex<double>(0, min_freq_f * tau_diff));
      for (int freq = 1; freq < num_freq_f; ++freq) {
        exp_f[k][l][freq] = rat * exp_f[k][l][freq - 1] * sign_mod;
      }

      exp_b[k][l][0] = 1.0;
      for (int freq = 1; freq < num_freq_b; ++freq) {
        exp_b[k][l][freq] = rat * exp_b[k][l][freq - 1];
      }
    }
  }

  double sum_wprime = 0.0;
  auto extents = boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][num_freq_f][num_freq_f][num_freq_b];
  boost::multi_array<std::complex<double>,7> result_tmp(extents);
  std::fill(result_tmp.origin(), result_tmp.origin() + result_tmp.num_elements(), 0.0);
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
          // Here, wprime does not include signs arising from mod beta.
          SCALAR wprime = mc_config.sign * reconnect.weight_rat_intermediate_state() * tmp_mat.determinant();
          sum_wprime += std::abs(wprime);

          if (wprime == 0.0) {
            continue;
          }

          double rw_corr =
                  GWorm<2>::get_weight_correction(
                  annihilation_ops[a].time().time(),
                  creation_ops[b].time().time(),
                  annihilation_ops[c].time().time(),
                  creation_ops[d].time().time(),
                  mc_config.p_irbasis
                  );

          auto fa = annihilation_ops[a].flavor();
          auto fb = creation_ops[b].flavor();
          auto fc = annihilation_ops[c].flavor();
          auto fd = creation_ops[d].flavor();
          for (int freq_f1 = 0; freq_f1 < num_freq_f; ++freq_f1) {
            for (int freq_f2 = 0; freq_f2 < num_freq_f; ++freq_f2) {
              for (int freq_b = 0; freq_b < num_freq_b; ++freq_b) {
                  result_tmp[fa][fb][fc][fd][freq_f1][freq_f2][freq_b] += exp_f[a][b][freq_f1] * exp_f[c][d][freq_f2] * exp_b[a][d][freq_b] / rw_corr;
              }
            }
          }
        }
      }
    }
  }

  // Multiplied by 1/(sum_wprime * beta)
  // The origin of "beta" term is the same as G1.
  auto it2 = result.origin();
  for (auto it=result_tmp.origin(); it != result_tmp.origin() + result_tmp.num_elements(); ++it) {
    *it2 += (*it)/(sum_wprime * beta);
    ++it2;
  }
}

template<typename SCALAR>
void G1Measurement<SCALAR>::measure_via_hyb(const MonteCarloConfiguration<SCALAR> &mc_config,
                                           const IRbasis& basis,
                                           alps::accumulators::accumulator_set &measurements,
                                           alps::random01 &random,
                                           int max_num_ops,
                                           double eps) {

  Reconnections<SCALAR> reconnection(mc_config, random, max_num_ops, 1);

  {
    std::vector<double> histogram(mc_config.p_irbasis->bin_edges().size()-1, 0.0);
    auto p_worm_G1 = dynamic_cast<const GWorm<1>* >(mc_config.p_worm.get());
    histogram[p_worm_G1->get_bin_index()] = 1;
    measurements[(str_ + "_bin_histogram").c_str()] << histogram;
  }

  compute_G1<SCALAR>(basis, mc_config, reconnection, data_, bin_data_);
  ++ num_data_;

  if (num_data_ == max_num_data_) {
    //pass the data to ALPS libraries
    std::transform(data_.origin(), data_.origin() + data_.num_elements(), data_.origin(),
                   std::bind2nd(std::divides<std::complex<double> >(), 1. * max_num_data_));
    measure_simple_vector_observable<std::complex<double> >(measurements, str_.c_str(), to_std_vector(data_));

    std::transform(bin_data_.origin(), bin_data_.origin() + bin_data_.num_elements(), bin_data_.origin(),
                   std::bind2nd(std::divides<std::complex<double> >(), 1. * max_num_data_));
    measure_simple_vector_observable<std::complex<double> >(measurements, (str_+"_bin").c_str(), to_std_vector(bin_data_));

    num_data_ = 0;
    std::fill(data_.origin(), data_.origin() + data_.num_elements(), 0.0);
    std::fill(bin_data_.origin(), bin_data_.origin() + bin_data_.num_elements(), 0.0);
  }
}

template<typename SCALAR>
void G2Measurement<SCALAR>::measure_via_hyb(const MonteCarloConfiguration<SCALAR> &mc_config,
                                            const IRbasis& basis,
                                            alps::accumulators::accumulator_set &measurements,
                                            alps::random01 &random,
                                            int max_num_ops,
                                            double eps) {

  Reconnections<SCALAR> reconnection(mc_config, random, max_num_ops, 2);

  compute_G2<SCALAR>(basis, num_freq_f_, num_freq_b_, mc_config, reconnection, matsubara_data_);
  ++ num_data_;

  if (num_data_ == max_num_data_) {
    //pass the data to ALPS libraries
    std::transform(matsubara_data_.origin(), matsubara_data_.origin() + matsubara_data_.num_elements(), matsubara_data_.origin(),
                   std::bind2nd(std::divides<std::complex<double> >(), 1. * max_num_data_));
    measure_simple_vector_observable<std::complex<double> >(measurements, (str_ + "_matsubara").c_str(), to_std_vector(matsubara_data_));

    num_data_ = 0;
    std::fill(matsubara_data_.origin(), matsubara_data_.origin() + matsubara_data_.num_elements(), 0.0);
  }
}

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
