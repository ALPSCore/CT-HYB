#include "./measurement.hpp"

#include <boost/functional/hash.hpp>

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

std::pair<double, double>
mod_beta(double tau, double beta) {
  if (tau > 0 && tau < beta) {
    return std::make_pair(tau, 1.0);
  } else if (tau < 0 && tau > -beta) {
    return std::make_pair(tau + beta, -1.0);
  } else {
    throw std::runtime_error("error in mod_beta!");
  }
};

int
mod_sign(double tau, double beta) {
  if (tau > 0 && tau < beta) {
    return 1;
  } else if (tau < 0 && tau > -beta) {
    return -1;
  } else {
    throw std::runtime_error("error in mod_sign!");
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
  for (int i = 0; i < pert_order; ++i) {
    for (int j = 0; j < pert_order; ++j) {
      M_(i, j) = 0;
    }
  }
  int offset = 0;
  for (int ib = 0; ib < mc_config.M.num_blocks(); ++ib) {
    const int block_size = mc_config.M.block_matrix_size(ib);
    M_.block(offset, offset, block_size, block_size) = mc_config.M.compute_inverse_matrix(ib);
    offset += block_size;
  }
  matrix_t B(pert_order, Rank + n_aux_lines), C(Rank + n_aux_lines, pert_order), D(Rank + n_aux_lines,
                                                                                   Rank + n_aux_lines);
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
          ++i_reduced;
        }
        ++j_reduced;
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
                const MonteCarloConfiguration<SCALAR> &mc_config,
                const Reconnections<SCALAR> &reconnect,
                boost::multi_array<std::complex<double>, 3> &result,
                boost::multi_array<std::complex<double>, 3> &bin_result
) {
  double beta = basis.beta();

  std::vector<double> Ultau_vals(static_cast<int>(basis.dim_F()));
  const auto &M = reconnect.M();
  const auto &creation_ops = reconnect.creation_ops();
  const auto &annihilation_ops = reconnect.annihilation_ops();
  SCALAR sign = mc_config.sign;

  std::vector<psi>::const_iterator it1, it2;
  const int mat_size = M.size1();

  //First, we compute relative weights
  boost::multi_array<SCALAR, 2> wprimes(boost::extents[mat_size - 1][mat_size - 1]);
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
  double scale_fact = -1.0 / (sum_wprime * beta);
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
      auto bin_length = mc_config.p_irbasis->bin_edges()[idx + 1] - mc_config.p_irbasis->bin_edges()[idx];
      bin_result[flavor_a][flavor_c][idx] += coeff / bin_length;
    }
  }
}

template<typename SCALAR>
void measure_G2_k4_PH(
    double beta,
    int num_flavors,
    int num_phys_rows,
    SCALAR overall_coeff,
    const Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic>& M_prime,
    const std::vector<psi>& creation_ops,
    const std::vector<psi>& annihilation_ops,
    const std::vector<int>& freq_index_f,
    const std::vector<int>& freq_index_b,
    boost::multi_array<std::complex<double>, 7> &result
) {
  // List of fermion frequencies
  auto iwn_f_min = freq_index_f[0];
  double min_freq_f = M_PI * (2 * iwn_f_min + 1) / beta;
  auto num_freq_f = freq_index_f.size();
  auto num_freq_b = freq_index_b.size();

  boost::multi_array<std::complex<double>, 3>
      exp_f(boost::extents[num_phys_rows][num_phys_rows][num_freq_f]);//annihilator, creator, freq_f
  boost::multi_array<std::complex<double>, 3>
      exp_b(boost::extents[num_phys_rows][num_phys_rows][num_freq_b]);//annihilator, creator, freq_b
  double tau_diff, sign_mod;
  for (int k = 0; k < num_phys_rows; k++) {
    for (int l = 0; l < num_phys_rows; l++) {
      double argument = annihilation_ops[k].time() - creation_ops[l].time();

      std::tie(tau_diff, sign_mod) = mod_beta(argument, beta);

      for (int freq = 0; freq < num_freq_f; ++freq) {
        auto wn =  M_PI * (2 * freq_index_f[freq] + 1) / beta;
        exp_f[k][l][freq] = sign_mod * std::exp(std::complex<double>(0, wn * tau_diff));
      }

      for (int freq = 0; freq < num_freq_b; ++freq) {
        auto wn =  M_PI * (2 * freq_index_b[freq]) / beta;
        exp_b[k][l][freq] = std::exp(std::complex<double>(0, wn * tau_diff));
      }
    }
  }

  auto extents = boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][num_freq_f][num_freq_f][num_freq_b];
  boost::multi_array<std::complex<double>, 7> result_tmp(extents);
  std::fill(result_tmp.origin(), result_tmp.origin() + result_tmp.num_elements(), 0.0);
  for (int a = 0; a < num_phys_rows; ++a) {
    for (int b = 0; b < num_phys_rows; ++b) {
      for (int c = 0; c < num_phys_rows; ++c) {
        if (a == c) {
          continue;
        }
        for (int d = 0; d < num_phys_rows; ++d) {
          if (b == d) {
            continue;
          }
          /*
           * Delta convention
           * M_ab  M_ad  M_a*
           * M_cb  M_cd  M_c*
           * M_*b  M_*d  M_**
           *
           * Here, M is in F convention. Rows and columns must be swapped.
           */
          SCALAR det = (M_prime(b,a) * M_prime(d,c) - M_prime(d,a) * M_prime(b,c));

          if (det == 0.0) {
            continue;
          }

          auto fa = annihilation_ops[a].flavor();
          auto fb = creation_ops[b].flavor();
          auto fc = annihilation_ops[c].flavor();
          auto fd = creation_ops[d].flavor();
          for (int freq_f1 = 0; freq_f1 < num_freq_f; ++freq_f1) {
            for (int freq_f2 = 0; freq_f2 < num_freq_f; ++freq_f2) {
              for (int freq_b = 0; freq_b < num_freq_b; ++freq_b) {
                result_tmp[fa][fb][fc][fd][freq_f1][freq_f2][freq_b] +=
                    det * exp_f[a][b][freq_f1] * exp_f[c][d][freq_f2] * exp_b[a][d][freq_b];
              }
            }
          }
        }
      }
    }
  }

  auto it2 = result.origin();
  for (auto it = result_tmp.origin(); it != result_tmp.origin() + result_tmp.num_elements(); ++it) {
    *it2 += (*it) * overall_coeff;
    ++it2;
  }
}

/*
 * Compute intermediate single-particle object for measuring two-particle GF
 * g_{f1,f2}(omega, omega') = sum_{ij} M'_{ji} exp(i omega tau_i - i omega' tau_j) delta_{f1, i} delta_{f2, j}
 */
template<typename SCALAR>
Eigen::Tensor<std::complex<double>,4>
    compute_g(
        double beta,
        int num_flavors,
        int num_phys_rows,
        const Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic>& M_prime,
        const std::vector<int>& index_freqs_f,
        const std::vector<psi>& creation_ops,
        const std::vector<psi>& annihilation_ops
    ) {

  using dcomplex = std::complex<double>;

  auto num_freqs = index_freqs_f.size();

  auto to_fermion_freq = [&](int n) {
      return (2 * n + 1) * M_PI / beta;
  };

  //std::cout << "debug a" << std::endl;
  std::vector<double> freqs_f(num_freqs);
  for (int i=0; i<num_freqs; ++i) {
    freqs_f[i] = to_fermion_freq(index_freqs_f[i]);
  }

  Eigen::Tensor<dcomplex,4> g(num_freqs, num_freqs, num_flavors, num_flavors);
  g.setZero();

  Eigen::Tensor<dcomplex,3> exp_1(num_freqs, 1, num_phys_rows), exp_2_conj(1, num_freqs, num_phys_rows);

  auto expix = [](double x) {
      return std::complex<double>(std::cos(x), std::sin(x));
  };

  for (int i=0; i<num_phys_rows; ++i) {
    auto tau_i = annihilation_ops[i].time().time();
    for (int f=0; f<num_freqs; ++f) {
      exp_1(f, 0, i) = expix(freqs_f[f] * tau_i);
    }
  }

  for (int j=0; j<num_phys_rows; ++j) {
    auto tau_j = creation_ops[j].time().time();
    for (int f=0; f<num_freqs; ++f) {
      exp_2_conj(0, f, j) = expix(-freqs_f[f] * tau_j);
    }
  }
  //std::cout << "debug b " << num_phys_rows << std::endl;

  // O( k^2 * num_freqs_f + k^2 * num_freqs_f^2 * num_flavors^2)
  // k is expansion order per flavor
  // Typical case:
  //    k = 10
  //    num_freqs_f = 10^3
  //    num_flavors = 2
  //  => 10^5 + 4 * 10^8
  auto max_num_ops = 0;
  for (int f_i = 0; f_i < num_flavors; ++f_i) {
    for (int f_j = 0; f_j < num_flavors; ++f_j) {
      int idx = 0;
      for (int i = 0; i < num_phys_rows; ++i) {
        if (annihilation_ops[i].flavor() != f_i) {
          continue;
        }
        for (int j = 0; j < num_phys_rows; ++j) {
          if (creation_ops[j].flavor() != f_j) {
            continue;
          }
          ++ idx;
        }
      }
      max_num_ops = std::max(max_num_ops, idx);
    }
  }

  Eigen::MatrixXcd exp_1_mat(num_freqs, max_num_ops), exp_2_conj_mat(num_freqs, max_num_ops), tmp_mat(num_freqs, num_freqs);
  for (int f_i = 0; f_i < num_flavors; ++f_i) {
    for (int f_j = 0; f_j < num_flavors; ++f_j) {

      int idx = 0;
      for (int i=0; i<num_phys_rows; ++i) {
        if (annihilation_ops[i].flavor() != f_i) {
          continue;
        }
        for (int j=0; j<num_phys_rows; ++j) {
          if (creation_ops[j].flavor() != f_j) {
            continue;
          }

          for (int f=0; f < num_freqs; ++f) {
            exp_1_mat(f, idx) = M_prime(j,i) * exp_1(f, 0, i);
            exp_2_conj_mat(f, idx) = exp_2_conj(0, f, j);
          }

          ++idx;
        }//i
      }//j

      if (idx > 0) {
        //std::cout << " f_i " << f_i  << " " << f_j << " " << idx << std::endl;
        tmp_mat = exp_1_mat.block(0,0,num_freqs,idx) * exp_2_conj_mat.block(0,0,num_freqs,idx).transpose();
        //std::cout << " f_i " << f_i  << " " << f_j << " " << idx << std::endl;
        g.chip(f_j, 3).chip(f_i, 2) = Eigen::TensorMap<Eigen::Tensor<dcomplex,2>>(&tmp_mat(0,0), num_freqs, num_freqs);
        //std::cout << " f_i " << f_i  << " " << f_j << " " << idx << std::endl;
      }
    }
  }

  //std::cout << "debug c " << num_phys_rows << std::endl;

  return g;
}

template<typename SCALAR, typename MAT>
void
compute_g_flavor(
    double beta,
    const MAT& M_prime,
    const std::vector<std::pair<int,int>>& two_freqs,
    const Eigen::Tensor<std::complex<double>,2>& exp_creation_ops,
    const Eigen::Tensor<std::complex<double>,2>& exp_annihilation_ops,
    std::complex<double>* output_buffer
) {
  using dcomplex = std::complex<double>;

  if (exp_creation_ops.size() * exp_annihilation_ops.size() == 0) {
    return;
  }

  /*
  *output_buffer = 0;
  for (int j=0; j<creation_ops.size(); ++j) {
    for (int i=0; i<annihilation_ops.size(); ++i) {
      for (int f = 0; f < two_freqs.size(); ++f) {
        *(output_buffer + f) += M_prime(j, i) * exp_annihilation_ops(f, i) * exp_creation_ops(f, j).conjg();
      }
    }//i
  }//j
   */

  //return g;
}

template<typename SCALAR>
Eigen::Tensor<std::complex<double>,3>
compute_g_new(
    double beta,
    int num_flavors,
    int num_phys_rows,
    const Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic>& M_prime,
    const std::vector<std::pair<int,int>>& two_freqs,
    const std::vector<psi>& creation_ops,
    const std::vector<psi>& annihilation_ops
) {

  using dcomplex = std::complex<double>;

  auto to_fermion_freq = [&](int n) {
      return (2 * n + 1) * M_PI / beta;
  };

  //std::vector<double> freqs_f(num_freqs);
  //for (int i=0; i<num_freqs; ++i) {
    //freqs_f[i] = to_fermion_freq(index_freqs_f[i]);
  //}

  Eigen::Tensor<dcomplex,3> g(two_freqs.size(), num_flavors, num_flavors);
  g.setZero();

  auto expix = [](double x) {
      return std::complex<double>(std::cos(x), std::sin(x));
  };

  auto pick_up_one_flavor = [&](const std::vector<psi>& ops, int flavor, const std::vector<int>& freqs) {
    std::vector<int> pos;
    for (int i=0; i<ops.size(); ++i) {
      if (ops[i].flavor() == flavor) {
        pos.push_back(i);
      }
    }
    auto nr = pos.size();

    auto exp_tensor = Eigen::Tensor<dcomplex,2>(freqs.size(), nr);
    for (int i=0; i<nr; ++i) {
      auto p = pos[i];
      auto tau = ops[p].time().time();
      assert(ops[p].flavor() == flavor);
      for (int f=0; f<freqs.size(); ++f) {
        exp_tensor(f, i) = expix(to_fermion_freq(freqs[f]) * tau);
      }
    }

    return std::make_pair(pos, exp_tensor);
  };

  std::vector<Eigen::Tensor<dcomplex,2>> exp_cr(num_flavors), exp_ann(num_flavors);
  std::vector<std::vector<int>> cr_rows(num_flavors), ann_rows(num_flavors);
  std::vector<int> freqs_cr, freqs_ann;
  std::transform(two_freqs.begin(), two_freqs.end(), std::back_inserter(freqs_cr), [](std::pair<int,int> p) {return p.second;});
  std::transform(two_freqs.begin(), two_freqs.end(), std::back_inserter(freqs_ann), [](std::pair<int,int> p) {return p.first;});
  for (int flavor=0; flavor<num_flavors; ++flavor) {
    std::tie(cr_rows[flavor], exp_cr[flavor]) = pick_up_one_flavor(creation_ops, flavor, freqs_cr);
    std::tie(ann_rows[flavor], exp_ann[flavor]) = pick_up_one_flavor(annihilation_ops, flavor, freqs_ann);
  }

  for (int f_ann = 0; f_ann < num_flavors; ++f_ann) {
    auto n_ann = ann_rows[f_ann].size();
    if (n_ann == 0) {
      continue;
    }

    // O(k^2 * num_freqs * num_flavors^2)
    // Typical case:
    //    k = 10 (per flavor)
    //    num_freqs = 10^4
    //    num_flavors = 2
    //  => 4 x 10^6
    for (int f_cr = 0; f_cr < num_flavors; ++f_cr) {
      auto n_cr = cr_rows[f_cr].size();
      if (n_cr == 0) {
        continue;
      }


      Eigen::Tensor<dcomplex,2> M_prime_tmp(n_cr, n_ann);

      for (int i=0; i < n_ann; ++i) {
        for (int j=0; j < n_cr; ++j) {
          M_prime_tmp(j, i) = M_prime(cr_rows[f_cr][j], ann_rows[f_ann][i]);
        }
      }

      //M_prime(j,i) * exp_ann(f,i) => (j,f)
      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 1) };
      Eigen::Tensor<dcomplex,2> Mprime_exp_ann = M_prime_tmp.contract(exp_ann[f_ann], product_dims);

      // (j,f) * (f,j) => (f, j) => (f)
      std::array<int,2> shuffle {1, 0};
      Eigen::TensorMap<Eigen::Tensor<dcomplex,1>> obuff(&g(0, f_ann, f_cr), two_freqs.size());
      std::array<int,1> sum_dims {1};
      obuff = (Mprime_exp_ann.shuffle(shuffle) * exp_cr[f_cr].conjugate()).sum(sum_dims);

      //The following does the same job as the above code.
      /*
      for (int j=0; j<n_cr; ++j) {
        for (int i=0; i<n_ann; ++i) {
          for (int f = 0; f < two_freqs.size(); ++f) {
            g(f, f_ann, f_cr) += M_prime_tmp(j, i) * exp_ann[f_ann](f, i) * std::conj(exp_cr[f_cr](f, j));
          }
        }//i
      }//j
       */

    }
  }

  return g;
}

namespace detail {
    class HashIntPair
    {
    public:
        std::size_t operator()(const std::pair<int,int>& p) const
        {
          std::size_t seed = 0;
          boost::hash_combine(seed, p.first);
          boost::hash_combine(seed, p.second);
          return seed;
        }
    };
}


/**
 * Does the same job as measure_G2_k4_PH. But computational cost scales as O(k^2) where k is the matrix size for reconnection
 * @tparam SCALAR
 * @param beta
 * @param num_flavors
 * @param num_phys_rows
 * @param overall_coeff
 * @param M_prime
 * @param creation_ops
 * @param annihilation_ops
 * @param meas_freqs_list
 * @param result
 */
template<typename SCALAR, typename MULTI_ARRAY_DIM5>
void measure_G2_k2_PH_impl(
    double beta,
    int num_flavors,
    int num_phys_rows,
    SCALAR overall_coeff,
    const Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic>& M_prime,
    const std::vector<psi>& creation_ops,
    const std::vector<psi>& annihilation_ops,
    const std::vector<matsubara_freq_point_PH>& meas_freqs_list,
    MULTI_ARRAY_DIM5 &result
) {
  using dcomplex = std::complex<double>;

  auto t1 = std::chrono::system_clock::now();

  std::vector<std::pair<int,int>> two_freqs_vec;
  for (const auto& freq: meas_freqs_list) {
    auto freq_f1 = std::get<0>(freq);
    auto freq_f2 = std::get<1>(freq);
    auto freq_b = std::get<2>(freq);
    two_freqs_vec.emplace_back(freq_f1+freq_b, freq_f1);
    two_freqs_vec.emplace_back(freq_f2, freq_f2+freq_b);
    two_freqs_vec.emplace_back(freq_f1+freq_b, freq_f2+freq_b);
    two_freqs_vec.emplace_back(freq_f2, freq_f1);
  }
  std::sort(two_freqs_vec.begin(), two_freqs_vec.end());
  std::unordered_map<std::pair<int,int>, int, detail::HashIntPair> two_freqs_map;
  for (int f = 0; f < two_freqs_vec.size(); ++f) {
    two_freqs_map.emplace(two_freqs_vec[f], f);
  }

  auto t2 = std::chrono::system_clock::now();

  Eigen::Tensor<std::complex<double>,3> g = compute_g_new(
      beta, num_flavors, num_phys_rows,
      M_prime, two_freqs_vec,
      creation_ops, annihilation_ops);

  auto t3 = std::chrono::system_clock::now();

  // g (freq, flavor, flavor) => // g_trans (flavor, flavor, freq)
  Eigen::array<int, 3> shuffle {1, 2, 0};
  Eigen::Tensor<std::complex<double>,3> g_trans = g.shuffle(shuffle);

  auto find_freq_index = [&](int freq1, int freq2) {
      auto idx = two_freqs_map[std::make_pair(freq1,freq2)];
      assert(idx >= 0 && idx < two_freqs_vec.size());
      if(idx < 0 || idx >= two_freqs_vec.size()) {
        throw std::logic_error("Something went wrong in find_freq_index.");
      }
      return idx;
  };

  // Assembly: O(num of freqs * N_f**4)
  auto nfreqs = meas_freqs_list.size();
  Eigen::Tensor<dcomplex,3> g1_H(num_flavors, num_flavors, nfreqs);
  Eigen::Tensor<dcomplex,3> g2_H(num_flavors, num_flavors, nfreqs);
  Eigen::Tensor<dcomplex,3> g1_F(num_flavors, num_flavors, nfreqs);
  Eigen::Tensor<dcomplex,3> g2_F(num_flavors, num_flavors, nfreqs);

  auto copy_elem = [&](int src_freq_idx,  Eigen::Tensor<dcomplex,3>& g1_out, int dst_freq_idx) {
      std::copy(
          &g_trans(0, 0, src_freq_idx),
          &g_trans(0, 0, src_freq_idx) + num_flavors * num_flavors,
          &g1_out(0, 0, dst_freq_idx)
      );
      return;
  };

  auto idx = 0;
  for (const auto& freq: meas_freqs_list) {
    auto freq_f1 = std::get<0>(freq);
    auto freq_f2 = std::get<1>(freq);
    auto freq_b = std::get<2>(freq);

    copy_elem(find_freq_index(freq_f1+freq_b, freq_f1), g1_H, idx);
    copy_elem(find_freq_index(freq_f2, freq_f2+freq_b), g2_H, idx);
    copy_elem(find_freq_index(freq_f1+freq_b, freq_f2+freq_b), g1_F, idx);
    copy_elem(find_freq_index(freq_f2, freq_f1), g2_F, idx);

    ++ idx;
  }

  for(int f1=0; f1<num_flavors; ++f1) {
    for (int f2 = 0; f2 < num_flavors; ++f2) {
      for (int f3 = 0; f3 < num_flavors; ++f3) {
        for (int f4 = 0; f4 < num_flavors; ++f4) {
          for (auto idx = 0; idx < nfreqs; ++idx) {
            result[f1][f2][f3][f4][idx] += overall_coeff * (g1_H(f1, f2, idx) * g2_H(f3, f4, idx) - g1_F(f1, f4, idx) * g2_F(f3, f2, idx));
          }
        }
      }
    }
  }

  auto t4 = std::chrono::system_clock::now();

  std::cout << " time "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count() << "  "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << "  "
                                                                                       << std::endl;
};

/**
 * Does the same job as measure_G2_k4_PH. But computational cost scales as O(k^2) where k is the matrix size for reconnection
 * @tparam SCALAR
 * @param beta
 * @param num_flavors
 * @param num_phys_rows
 * @param overall_coeff
 * @param M_prime
 * @param creation_ops
 * @param annihilation_ops
 * @param freq_index_f
 * @param freq_index_b
 * @param result
 */
template<typename SCALAR>
void measure_G2_k2_PH(
    double beta,
    int num_flavors,
    int num_phys_rows,
    SCALAR overall_coeff,
    const Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic>& M_prime,
    const std::vector<psi>& creation_ops,
    const std::vector<psi>& annihilation_ops,
    const std::vector<int>& freq_index_f,
    const std::vector<int>& freq_index_b,
    boost::multi_array<std::complex<double>, 7> &result
) {
  using dcomplex = std::complex<double>;

  // List of fermion frequencies
  auto num_freq_f = freq_index_f.size();
  auto num_freq_b = freq_index_b.size();

  std::vector<matsubara_freq_point_PH> freqs(num_freq_f * num_freq_f * num_freq_b);

  auto idx = 0;
  for (auto idx_f1 = 0; idx_f1<num_freq_f; ++idx_f1) {
    for (auto idx_f2 = 0; idx_f2<num_freq_f; ++idx_f2) {
      for (auto idx_b = 0; idx_b<num_freq_b; ++idx_b) {
        freqs[idx] = matsubara_freq_point_PH(freq_index_f[idx_f1], freq_index_f[idx_f2], freq_index_b[idx_b]);
        ++ idx;
      }
    }
  }

  auto extents_out = boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][freqs.size()];
  boost::multi_array_ref<dcomplex,5> out_buffer(result.origin(), extents_out);
  measure_G2_k2_PH_impl(beta, num_flavors, num_phys_rows, overall_coeff, M_prime,
      creation_ops, annihilation_ops, freqs, out_buffer);
}


//Compute G2 by removal in G2 space
template<typename SCALAR>
void compute_G2(const IRbasis &basis,
                const std::vector<matsubara_freq_point_PH>& freqs,
                const MonteCarloConfiguration<SCALAR> &mc_config,
                const Reconnections<SCALAR> &reconnect,
                boost::multi_array<std::complex<double>, 5> &result
) {
  double beta = basis.beta();

  const auto &M = reconnect.M();
  const auto &creation_ops = reconnect.creation_ops();
  const auto &annihilation_ops = reconnect.annihilation_ops();
  SCALAR sign = mc_config.sign;

  const int mat_size = M.size1();

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
  Eigen::Matrix<SCALAR, det_size, det_size> tmp_mat;
  boost::array<int, det_size> rows3, cols3;
  const int last = M.size1() - 1;
  for (int i = 0; i < n_aux_lines; ++i) {
    cols3[rank + i] = rows3[rank + i] = i + M.size1() - n_aux_lines;
  }
  assert(cols3.back() == last);

  /*
   * M = (A B)
   *     (C D)
   * |M| = |D| |A - BD^{-1} C| = |D| |M_prime|,
   * where M_prime = A - B D^{-1} C.
   */
  using matrix_type = Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic>;
  matrix_type A = M.block(0, 0, num_phys_rows, num_phys_rows);
  matrix_type B = M.block(0, num_phys_rows, num_phys_rows, n_aux_lines);
  matrix_type C = M.block(num_phys_rows, 0, n_aux_lines, num_phys_rows);
  matrix_type D = M.block(num_phys_rows, num_phys_rows, n_aux_lines, n_aux_lines);
  SCALAR det_D = D.determinant();
  matrix_type M_prime = A - B * D.inverse() * C;

  double sum_wprime = 0.0;
  for (int a = 0; a < num_phys_rows; ++a) {
    for (int b = 0; b < num_phys_rows; ++b) {
      for (int c = 0; c < num_phys_rows; ++c) {
        if (a == c) {
          continue;
        }
        for (int d = 0; d < num_phys_rows; ++d) {
          if (b == d) {
            continue;
          }
          /*
           * Delta convention
           * M_ab  M_ad  M_a*
           * M_cb  M_cd  M_c*
           * M_*b  M_*d  M_**
           *
           * Here, M is in F convention. Rows and columns must be swapped.
           */
          SCALAR det = det_D * (M_prime(b,a) * M_prime(d,c) - M_prime(d,a) * M_prime(b,c));

          double rw_corr =
              GWorm<2>::get_weight_correction(
                  annihilation_ops[a].time().time(),
                  creation_ops[b].time().time(),
                  annihilation_ops[c].time().time(),
                  creation_ops[d].time().time(),
                  mc_config.p_irbasis
              );

          // Here, wprime does not include signs arising from mod beta.
          // wprime is the weight of a Monte Carlo state with a reweighting facter being included
          SCALAR w_org = mc_config.sign * reconnect.weight_rat_intermediate_state() * det;
          SCALAR wprime = w_org * rw_corr;
          sum_wprime += std::abs(wprime);
        }
      }
    }
  }

  // Multiplied by 1/(sum_wprime * beta)
  // The origin of "beta" term is the same as G1.
  SCALAR overall_coeff = mc_config.sign * reconnect.weight_rat_intermediate_state() * det_D/ (sum_wprime * beta);

  measure_G2_k2_PH_impl(beta, num_flavors, num_phys_rows, overall_coeff,
                   M_prime, creation_ops, annihilation_ops, freqs, result);
}

//Compute G2 by removal in G2 space
template<typename SCALAR>
void compute_G2_IR(const IRbasis &basis,
                   const MonteCarloConfiguration<SCALAR> &mc_config,
                   const Reconnections<SCALAR> &reconnect,
                   boost::multi_array<std::complex<double>, 5> &result
) {
  double beta = basis.beta();

  const auto &M = reconnect.M();
  const auto &creation_ops = reconnect.creation_ops();
  const auto &annihilation_ops = reconnect.annihilation_ops();
  SCALAR sign = mc_config.sign;

  const int mat_size = M.size1();

  //First, we compute relative weights
  const double temperature = 1. / beta;
  const int num_flavors = result.shape()[0];
  const int num_phys_rows = creation_ops.size();
  const int n_aux_lines = 2;
  if (creation_ops.size() != annihilation_ops.size() || creation_ops.size() != M.size1() - n_aux_lines) {
    throw std::runtime_error("Fatal error in compute_G2_IR");
  }

  //naive way to evaluate
  //The indices of M are reverted from (C. 24) of L. Boehnke (2011) because we're using the F convention here.

  //First, compute relative weights
  const int rank = 2;
  const int det_size = rank + n_aux_lines;
  Eigen::Matrix<SCALAR, det_size, det_size> tmp_mat;
  boost::array<int, det_size> rows3, cols3;
  const int last = M.size1() - 1;
  for (int i = 0; i < n_aux_lines; ++i) {
    cols3[rank + i] = rows3[rank + i] = i + M.size1() - n_aux_lines;
  }
  assert(cols3.back() == last);

  boost::multi_array<double,2> bubble_sign(boost::extents[num_phys_rows][num_phys_rows]),
      tau_diff(boost::extents[num_phys_rows][num_phys_rows]);
  for (int k = 0; k < num_phys_rows; k++) {
    for (int l = 0; l < num_phys_rows; l++) {
      double argument = annihilation_ops[k].time() - creation_ops[l].time();
      auto r = mod_beta(argument, beta);
      tau_diff[k][l] = r.first;
      bubble_sign[k][l] = r.second;
    }
  }

  double sum_wprime = 0.0;
  auto extents = boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][basis.num_bins_4pt()];
  boost::multi_array<std::complex<double>, 5> result_tmp(extents);
  std::fill(result_tmp.origin(), result_tmp.origin() + result_tmp.num_elements(), 0.0);
  for (int a = 0; a < num_phys_rows; ++a) {
    for (int b = 0; b < num_phys_rows; ++b) {
      for (int c = 0; c < num_phys_rows; ++c) {
        if (a == c) {
          continue;
        }
        for (int d = 0; d < num_phys_rows; ++d) {
          if (b == d) {
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
              tmp_mat(i, j) = M(rows3[i], cols3[j]);
            }
          }


          // Here, wprime does not include signs arising from mod beta.
          // wprime is the weight of a Monte Carlo state with a reweighting facter being included
          // w_org is the Monte Carlo weight without reweighting factor.
          SCALAR w_org = mc_config.sign * reconnect.weight_rat_intermediate_state() * tmp_mat.determinant();

          if (w_org == 0.0) {
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
          SCALAR wprime = w_org * rw_corr;
          sum_wprime += std::abs(wprime);

          auto fa = annihilation_ops[a].flavor();
          auto fb = creation_ops[b].flavor();
          auto fc = annihilation_ops[c].flavor();
          auto fd = creation_ops[d].flavor();

          auto t1 = annihilation_ops[a].time().time();
          auto t2 = creation_ops[b].time().time();
          auto t3 = annihilation_ops[c].time().time();
          auto t4 = creation_ops[d].time().time();

          int bin_index = basis.get_bin_position_4pt(basis.get_index_4pt(t1, t2, t3, t4));
          SCALAR mod_signs = mod_sign(t1-t4, beta) * mod_sign(t2-t4, beta) * mod_sign(t3-t4, beta);

          // Here, we accumulate the two-particle GF "averaged" (not integrated) in bins.
          result_tmp[fa][fb][fc][fd][bin_index] += (w_org * mod_signs)/basis.bin_volume_4pt(bin_index);
        }
      }
    }
  }

  // Multiplied by 1/(sum_wprime * beta)
  // The origin of "beta" term is the same as G1.
  auto it2 = result.origin();
  for (auto it = result_tmp.origin(); it != result_tmp.origin() + result_tmp.num_elements(); ++it) {
    *it2 += (*it) / (sum_wprime * beta);
    ++it2;
  }
}

template<typename SCALAR>
void G1Measurement<SCALAR>::measure_via_hyb(const MonteCarloConfiguration<SCALAR> &mc_config,
                                            const IRbasis &basis,
                                            alps::accumulators::accumulator_set &measurements,
                                            alps::random01 &random,
                                            int max_num_ops,
                                            double eps) {

  Reconnections<SCALAR> reconnection(mc_config, random, max_num_ops, 1);

  {
    std::vector<double> histogram(mc_config.p_irbasis->bin_edges().size() - 1, 0.0);
    auto p_worm_G1 = dynamic_cast<const GWorm<1> * >(mc_config.p_worm.get());
    histogram[p_worm_G1->get_bin_index()] = 1;
    measurements[(str_ + "_bin_histogram").c_str()] << histogram;
  }

  compute_G1<SCALAR>(basis, mc_config, reconnection, data_, bin_data_);
  ++num_data_;

  if (num_data_ == max_num_data_) {
    //pass the data to ALPS libraries
    std::transform(data_.origin(), data_.origin() + data_.num_elements(), data_.origin(),
                   std::bind2nd(std::divides<std::complex<double> >(), 1. * max_num_data_));
    measure_simple_vector_observable<std::complex<double> >(measurements, str_.c_str(), to_std_vector(data_));

    std::transform(bin_data_.origin(), bin_data_.origin() + bin_data_.num_elements(), bin_data_.origin(),
                   std::bind2nd(std::divides<std::complex<double> >(), 1. * max_num_data_));
    measure_simple_vector_observable<std::complex<double> >(measurements, (str_ + "_bin").c_str(),
                                                            to_std_vector(bin_data_));

    num_data_ = 0;
    std::fill(data_.origin(), data_.origin() + data_.num_elements(), 0.0);
    std::fill(bin_data_.origin(), bin_data_.origin() + bin_data_.num_elements(), 0.0);
  }
}

template<typename SCALAR>
void G2Measurement<SCALAR>::measure_via_hyb(const MonteCarloConfiguration<SCALAR> &mc_config,
                                            const IRbasis &basis,
                                            alps::accumulators::accumulator_set &measurements,
                                            alps::random01 &random,
                                            int max_num_ops,
                                            double eps) {

  Reconnections<SCALAR> reconnection(mc_config, random, max_num_ops, 2);

  compute_G2<SCALAR>(basis, freqs_, mc_config, reconnection, matsubara_data_);
  ++num_data_;

  if (num_data_ == max_num_data_) {
    //pass the data to ALPS libraries
    std::transform(matsubara_data_.origin(), matsubara_data_.origin() + matsubara_data_.num_elements(),
                   matsubara_data_.origin(),
                   std::bind2nd(std::divides<std::complex<double> >(), 1. * max_num_data_));
    measure_simple_vector_observable<std::complex<double> >(measurements, (str_ + "_matsubara").c_str(),
                                                            to_std_vector(matsubara_data_));

    num_data_ = 0;
    std::fill(matsubara_data_.origin(), matsubara_data_.origin() + matsubara_data_.num_elements(), 0.0);
  }
}

template<typename SCALAR>
void G2IRMeasurement<SCALAR>::measure_via_hyb(const MonteCarloConfiguration<SCALAR> &mc_config,
                                            const IRbasis &basis,
                                            alps::accumulators::accumulator_set &measurements,
                                            alps::random01 &random,
                                            int max_num_ops,
                                            double eps) {

  Reconnections<SCALAR> reconnection(mc_config, random, max_num_ops, 2);

  compute_G2_IR<SCALAR>(basis, mc_config, reconnection, data_);
  ++num_data_;

  if (num_data_ == max_num_data_) {
    //pass the data to ALPS libraries
    std::transform(data_.origin(), data_.origin() + data_.num_elements(),
                   data_.origin(),
                   std::bind2nd(std::divides<std::complex<double> >(), 1. * max_num_data_));
    measure_simple_vector_observable<std::complex<double> >(measurements, (str_ + "_bin").c_str(),
                                                            to_std_vector(data_));

    num_data_ = 0;
    std::fill(data_.origin(), data_.origin() + data_.num_elements(), 0.0);
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
