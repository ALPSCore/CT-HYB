#include "./ortho_measurement.hpp"

template<typename SCALAR, typename SW_TYPE, int Rank>
void GOrthoBasisMeasurement<SCALAR, SW_TYPE, Rank>::measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements) {
  typedef typename ExtendedScalar<SCALAR>::value_type EXTENDED_SCALAR;
  typedef operator_container_t::iterator Iterator;
  typedef Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

  const int pert_order = mc_config.pert_order();
  std::shared_ptr<HybridizationFunction<SCALAR> > p_gf = mc_config.M.get_greens_function();
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
        D(i, j) = eps_ * random();
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
  if (pert_order + Rank > max_num_ops_) {
    const int num_ops = pert_order + Rank;
    std::vector<bool> is_row_active(num_ops + n_aux_lines, false), is_col_active(num_ops + n_aux_lines, false);
    //always choose the original worm position for detailed balance condition
    for (int i = 0; i < Rank + n_aux_lines; ++i) {
      //is_row_active[num_ops - i] = true;
      //is_col_active[num_ops - i] = true;
      is_row_active[is_row_active.size() - 1 - i] = true;
      is_col_active[is_col_active.size() - 1 - i] = true;
    }
    for (int i = 0; i < max_num_ops_ - Rank; ++i) {
      is_row_active[i] = true;
      is_col_active[i] = true;
    }
    MyRandomNumberGenerator rnd(*p_rng_);
    std::random_shuffle(is_row_active.begin(), is_row_active.begin() + pert_order, rnd);
    std::random_shuffle(is_col_active.begin(), is_col_active.begin() + pert_order, rnd);
    assert(boost::count(is_col_active, true) == max_num_ops_ + n_aux_lines);
    assert(boost::count(is_row_active, true) == max_num_ops_ + n_aux_lines);

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
      assert(cdagg_ops_new.size() == max_num_ops_);
      assert(c_ops_new.size() == max_num_ops_);
    }

    {
      const int mat_size = M.size1();
      alps::fastupdate::ResizableMatrix<SCALAR> M_reduced(max_num_ops_ + n_aux_lines, max_num_ops_ + n_aux_lines, 0.0);
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
        assert(i_reduced == max_num_ops_ + n_aux_lines);
      }
      assert(j_reduced == max_num_ops_ + n_aux_lines);
      std::swap(M, M_reduced);
      assert(M.size1() == max_num_ops_ + n_aux_lines);
      assert(M.size2() == max_num_ops_ + n_aux_lines);
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
                                        p_basis_,
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

//Measure G1 by removal in G1 space
template<typename SCALAR>
int MeasureGHelper<SCALAR, 1>::perform(double beta,
                                        std::shared_ptr<OrthogonalBasis> p_basis,
                                        int n_freq,
                                        SCALAR sign,
                                        SCALAR weight_rat_intermediate_state,
                                        const std::vector<psi> &creation_ops,
                                        const std::vector<psi> &annihilation_ops,
                                        const alps::fastupdate::ResizableMatrix<SCALAR> &M,
                                        boost::multi_array<std::complex<double>, 3> &result) {
  const double temperature = 1. / beta;
  const int num_flavors = result.shape()[0];
  const int basis_dim = p_basis->dim();
  assert(result.shape()[2] == basis_dim);
  p_basis->sanity_check();

  std::vector<double> Ul_vals(basis_dim);
  std::vector<double> inv_norm2(basis_dim);
  for (auto l=0; l<basis_dim; ++l) {
    inv_norm2[l] = 1/p_basis->norm2(l);
  }

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

  if (norm == 0) {
    return 0;
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
      assert(0 <= argument && argument <= beta);

      const int flavor_a = it1->flavor();
      const int flavor_c = it2->flavor();
      p_basis->value(argument, Ul_vals);
      for (int il = 0; il < basis_dim; ++il) {
        result[flavor_a][flavor_c][il] += scale_fact * coeffs[k][l] * inv_norm2[il] * Ul_vals[il];
      }
    }
  }
  return 1;
};

//Measure G2 by removal in G2 space
template<typename SCALAR>
int MeasureGHelper<SCALAR, 2>::perform(double beta,
                                        std::shared_ptr<OrthogonalBasis> p_basis,
                                        int n_freq,
                                        SCALAR sign,
                                        SCALAR weight_rat_intermediate_state,
                                        const std::vector<psi> &creation_ops,
                                        const std::vector<psi> &annihilation_ops,
                                        const alps::fastupdate::ResizableMatrix<SCALAR> &M,
                                        boost::multi_array<std::complex<double>, 7> &result) {
  const double temperature = 1. / beta;
  const int num_flavors = result.shape()[0];
  const int basis_dim = p_basis->dim();
  const int num_phys_rows = creation_ops.size();
  const int n_aux_lines = 2;
  if (creation_ops.size() != annihilation_ops.size() || creation_ops.size() != M.size1() - n_aux_lines) {
    throw std::runtime_error("Fatal error in MeasureGHelper<SCALAR, 2>::perform()");
  }

  //Compute values of P
  std::vector<double> inv_norm2(basis_dim);
  for (auto l=0; l<basis_dim; ++l){
    inv_norm2[l] = 1/p_basis->norm2(l);
  }
  std::vector<double> inv_norm2_p(inv_norm2);
  for (int il = 0; il < basis_dim; il += 2) {
    inv_norm2_p[il] *= -1;
  }

  boost::multi_array<double, 3>
      norm2_Ul(boost::extents[num_phys_rows][num_phys_rows][basis_dim]);//annihilator, creator, legendre
  boost::multi_array<double, 3>
      norm2_Ul_p(boost::extents[num_phys_rows][num_phys_rows][basis_dim]);//annihilator, creator, legendre
  {
    std::vector<double> Ul_value(basis_dim);
    for (int k = 0; k < num_phys_rows; k++) {
      for (int l = 0; l < num_phys_rows; l++) {
        double argument = annihilation_ops[k].time() - creation_ops[l].time();
        double arg_sign = 1.0;
        if (argument < 0) {
          argument += beta;
          arg_sign = -1.0;
        }
        p_basis->value(argument, Ul_value);
        for (int il = 0; il < basis_dim; ++il) {
          norm2_Ul[k][l][il] = arg_sign * Ul_value[il] * inv_norm2[il];
          norm2_Ul_p[k][l][il] = arg_sign * Ul_value[il] * inv_norm2_p[il];
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
  if (norm == 0) {
    return 0;
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
          for (int il = 0; il < basis_dim; ++il) {
            for (int il_p = 0; il_p < basis_dim; ++il_p) {
              const SCALAR coeff2 = coeff * norm2_Ul[a][b][il] * norm2_Ul_p[c][d][il_p];
              for (int im = 0; im < n_freq; ++im) {
                result[flavor_a][flavor_b][flavor_c][flavor_d][il][il_p][im] += coeff2 * expiomega[a][d][im];
              }
            }
          }
        }
      }
    }
  }
  return 1;
};