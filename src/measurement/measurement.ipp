#include "./measurement.hpp"

#include <boost/functional/hash.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/tuple.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/multi_array.hpp>


inline
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

inline
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
  std::shared_ptr<HybridizationFunction<SCALAR> > p_gf = mc_config.M.get_greens_function();
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

  return g;
}

/*
 * Compute intermediate single-particle object for measuring two-particle GF
 * g_{f1,f2}(omega, omega') = sum_{ij} M'_{ji} exp(i omega tau_i - i omega' tau_j) delta_{f1, i} delta_{f2, j}
 */
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

  double M_prime_abs_sum = M_prime.cwiseAbs().sum();

  std::vector<Eigen::Tensor<dcomplex,2>> exp_cr(num_flavors), exp_ann(num_flavors);
  std::vector<std::vector<int>> cr_rows(num_flavors), ann_rows(num_flavors);
  std::vector<int> freqs_cr, freqs_ann;
  std::transform(two_freqs.begin(), two_freqs.end(), std::back_inserter(freqs_cr), [](std::pair<int,int> p) {return p.second;});
  std::transform(two_freqs.begin(), two_freqs.end(), std::back_inserter(freqs_ann), [](std::pair<int,int> p) {return p.first;});
  for (int flavor=0; flavor<num_flavors; ++flavor) {
    std::tie(cr_rows[flavor], exp_cr[flavor]) = pick_up_one_flavor(creation_ops, flavor, freqs_cr);
    std::tie(ann_rows[flavor], exp_ann[flavor]) = pick_up_one_flavor(annihilation_ops, flavor, freqs_ann);
  }

  // O(k^2 * num_freqs * num_flavors^2)
  // Typical case:
  //    k = 10 (per flavor)
  //    num_freqs = 10^4
  //    num_flavors = 2
  //  => 4 x 10^6
  for (int f_ann = 0; f_ann < num_flavors; ++f_ann) {
    auto n_ann = ann_rows[f_ann].size();
    if (n_ann == 0) {
      continue;
    }
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

      double abs_sum = Eigen::TensorFixedSize<double, Eigen::Sizes<>>(M_prime_tmp.abs().sum())(0);
      if (abs_sum < 1e-8 * M_prime_abs_sum) {
        continue;
      }

      //M_prime(k, kp) * exp_ann(freq, kp) => (k, freq)
      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 1) };
      Eigen::Tensor<dcomplex,2> Mprime_exp_ann = M_prime_tmp.contract(exp_ann[f_ann], product_dims);

      // [M_prime * exp_ann](k, freq) * exp_cr(freq, k) => (freq)
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



/**
 * Measure two-particle Green's function
 * Computational cost scales as O(k^2) where k is the matrix size for reconnection.
 * Only straight diagrams (denoted as [H}artree) is measured.
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
    const std::vector<std::pair<int,int>>& two_freqs_vec,
    const std::unordered_map<std::pair<int,int>, int>& two_freqs_map,
    MULTI_ARRAY_DIM5 &result
) {
  using dcomplex = std::complex<double>;

  auto t1 = std::chrono::system_clock::now();

  // two_freqs_vec contains a list of fermionic frequencies of one-particle-GF-like object
  Eigen::Tensor<std::complex<double>,3> g = compute_g_new(
      beta, num_flavors, num_phys_rows,
      M_prime, two_freqs_vec,
      creation_ops, annihilation_ops);

  auto t2 = std::chrono::system_clock::now();

  // g (freq, flavor, flavor) => // g_trans (flavor, flavor, freq)
  Eigen::array<int, 3> shuffle {1, 2, 0};
  Eigen::Tensor<std::complex<double>,3> g_trans = g.shuffle(shuffle);

  auto t3 = std::chrono::system_clock::now();

  // Assembly: O(num of freqs * N_f**4)
  auto nfreqs = meas_freqs_list.size();
  auto find_freq_index = [&](int freq1, int freq2) {
      auto key = std::make_pair(freq1,freq2);
      return two_freqs_map.at(key);
  };
  for (auto ifreq = 0; ifreq < nfreqs; ++ifreq) {
    auto freq = meas_freqs_list[ifreq];
    auto freq_f1 = std::get<0>(freq);
    auto freq_f2 = std::get<1>(freq);
    auto freq_b = std::get<2>(freq);
    auto idx_two_freq1 = find_freq_index(freq_f1+freq_b, freq_f1);
    auto idx_two_freq2 = find_freq_index(freq_f2, freq_f2+freq_b);

    for(int f1 = 0; f1 < num_flavors; ++f1) {
      for (int f2 = 0; f2 < num_flavors; ++f2) {
        for (int f3 = 0; f3 < num_flavors; ++f3) {
          for (int f4 = 0; f4 < num_flavors; ++f4) {
            result[ifreq][f1][f2][f3][f4] += overall_coeff * g_trans(f1, f2, idx_two_freq1) * g_trans(f3, f4, idx_two_freq2);
          }
        }
      }
    }
  }

  auto t4 = std::chrono::system_clock::now();

  //std::cout << " time "
            //<< std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " "
            //<< std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count() << "  "
            //<< std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << "  "
                                                                                       //<< " k = " << M_prime.rows()
                                                                                       //<< std::endl;
};



//Compute G2 by removal in G2 space
template<typename SCALAR>
void compute_G2(double beta,
                int num_flavors,
                const std::vector<matsubara_freq_point_PH>& freqs,
                const std::vector<std::pair<int,int>>& two_freqs_vec,
                const std::unordered_map<std::pair<int,int>, int>& two_freqs_map,
                const MonteCarloConfiguration<SCALAR> &mc_config,
                const Reconnections<SCALAR> &reconnect,
                boost::multi_array<std::complex<double>, 5> &result
) {
  const auto &M = reconnect.M();
  const auto &creation_ops = reconnect.creation_ops();
  const auto &annihilation_ops = reconnect.annihilation_ops();
  SCALAR sign = mc_config.sign;

  const int mat_size = M.size1();

  //First, we compute relative weights
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

          // Here, wprime does not include signs arising from mod beta.
          SCALAR w_org = mc_config.sign * reconnect.weight_rat_intermediate_state() * det;
          sum_wprime += std::abs(w_org);
        }
      }
    }
  }

  // Multiplied by 1/(sum_wprime * beta)
  // The origin of "beta" term is the same as G1.
  SCALAR overall_coeff = mc_config.sign * reconnect.weight_rat_intermediate_state() * det_D/ (sum_wprime * beta);

  measure_G2_k2_PH_impl(beta, num_flavors, num_phys_rows, overall_coeff,
                   M_prime, creation_ops, annihilation_ops, freqs, two_freqs_vec, two_freqs_map, result);
}

template<typename SCALAR, typename SW_TYPE>
G2Measurement<SCALAR,SW_TYPE>::G2Measurement(alps::random01 *p_rng, double beta, int num_flavors,
    const std::vector<matsubara_freq_point_PH>& freqs,
    int max_matrix_size, double eps):
    p_rng_(p_rng),
    beta_(beta),
    num_flavors_(num_flavors),
    freqs_(),
    str_("G2"),
    max_matrix_size_(max_matrix_size),
    eps_(eps),
    num_data_(0) {

  // Make sure that Fock term can be reconstructed.
  std::set<matsubara_freq_point_PH > freqs_PH_set;
  auto add = [&](matsubara_freq_point_PH key) {
      if (freqs_PH_set.find(key) == freqs_PH_set.end()) {
        freqs_.push_back(key);
        freqs_PH_set.insert(key);
      };
  };
  for (auto freq_PH : freqs) {
    add(freq_PH);
    add(from_H_to_F(freq_PH));
  }

  make_two_freqs_list(freqs_, two_freqs_vec_, two_freqs_map_);

  matsubara_data_.resize(boost::extents[freqs_.size()][num_flavors][num_flavors][num_flavors][num_flavors]);
  std::fill(matsubara_data_.origin(), matsubara_data_.origin() + matsubara_data_.num_elements(), 0);
};

template<typename SCALAR, typename SW_TYPE>
void G2Measurement<SCALAR,SW_TYPE>::measure(const MonteCarloConfiguration<SCALAR> &mc_config,
    const SW_TYPE &sliding_window,
    alps::accumulators::accumulator_set &measurements) {
  auto t1 = std::chrono::system_clock::now();
  Reconnections<SCALAR> reconnection(mc_config, *p_rng_, max_matrix_size_, 2, eps_);
  auto t2 = std::chrono::system_clock::now();

  compute_G2<SCALAR>(beta_, num_flavors_, freqs_, two_freqs_vec_, two_freqs_map_, mc_config, reconnection, matsubara_data_);
  auto t3 = std::chrono::system_clock::now();
  ++num_data_;
}

template<typename SCALAR, typename SW_TYPE>
void G2Measurement<SCALAR, SW_TYPE>::save_results(const std::string& filename) {
  alps::mpi::communicator comm;

  comm.barrier();
  int num_tot_data;
  MPI_Reduce(
      &num_data_,
      &num_tot_data,
      1,
      alps::mpi::get_mpi_datatype(num_data_),
      MPI_SUM,
      0,
      comm
  );

  if (num_data_ == 0) {
    std::cout << "Warning G2 has not been measured on node " << comm.rank() << "! This may cause a problem such as dead lock. Run longer!" << std::endl;
  }

  if (comm.rank() == 0) {
    MPI_Reduce(
        MPI_IN_PLACE,
        matsubara_data_.origin(),
        matsubara_data_.num_elements(),
        MPI_CXX_DOUBLE_COMPLEX,
        MPI_SUM,
        0,
        comm
    );
  } else {
    MPI_Reduce(
        matsubara_data_.origin(),
        matsubara_data_.origin(),
        matsubara_data_.num_elements(),
        MPI_CXX_DOUBLE_COMPLEX,
        MPI_SUM,
        0,
        comm
    );
  }

  if (comm.rank() == 0) {
    std::transform(matsubara_data_.origin(), matsubara_data_.origin() + matsubara_data_.num_elements(),
                   matsubara_data_.origin(),
                   std::bind2nd(std::divides<std::complex<double> >(), 1. * num_tot_data));

    alps::hdf5::archive oar(filename, "a");
    oar["/simulation/results/G2H_matsubara/data"] = matsubara_data_;
    boost::multi_array<int,2> freqs_tmp(boost::extents[freqs_.size()][3]);
    for (int i=0; i<freqs_.size(); ++i) {
      freqs_tmp[i][0] = std::get<0>(freqs_[i]);
      freqs_tmp[i][1] = std::get<1>(freqs_[i]);
      freqs_tmp[i][2] = std::get<2>(freqs_[i]);
    }
    oar["/simulation/results/G2H_matsubara/freqs_PH"] = freqs_tmp;
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
