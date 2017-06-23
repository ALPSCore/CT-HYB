#include "measurement.hpp"

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

  const int dim_basis = basis_.dim();
  std::vector<double> Pl_vals(dim_basis);

  basis_.value(2 * tdiff / beta_ - 1.0, Pl_vals);
  for (int il = 0; il < dim_basis; ++il) {
    data
    [flavors[0]]
    [flavors[1]]
    [flavors[2]]
    [flavors[3]][il] += weight *  Pl_vals[il] * sqrt(2.0/basis_.norm2(il));
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

  //compute effective matrix
  const int num_phys_lines = M.size1() - n_aux_lines;
  matrix_t M_eff =
      M.block(0, 0, num_phys_lines, num_phys_lines) -
          M.block(0, num_phys_lines, num_phys_lines, n_aux_lines)
              * M.block(num_phys_lines, num_phys_lines, n_aux_lines, n_aux_lines).inverse()
              * M.block(num_phys_lines, 0, n_aux_lines, num_phys_lines);

  //drop small values
  const double cutoff = 1.0e-10 * M_eff.cwiseAbs().maxCoeff();
  for (int j = 0; j < num_phys_lines; ++j) {
    for (int i = 0; i < num_phys_lines; ++i) {
      if (std::abs(M_eff(i, j)) < cutoff) {
        M_eff(i, j) = 0.0;
      }
    }
  }

  //measure by removal as we would do for the partition function expansion
  MeasureGHelper<SCALAR, Rank>::perform(beta_,
                                        p_basis_f_,
                                        p_basis_b_,
                                        mc_config.sign,
                                        weight_rat *
                                          M.block(num_phys_lines, num_phys_lines, n_aux_lines, n_aux_lines).determinant(),
                                        cdagg_ops_new,
                                        c_ops_new,
                                        M_eff,
                                        data_,
                                        trans_tensor_H_to_F_
  );
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
void MeasureGHelper<SCALAR, 1>::perform(double beta,
                                        boost::shared_ptr<OrthogonalBasis> p_basis_f,
                                        boost::shared_ptr<OrthogonalBasis> p_basis_b,
                                        SCALAR sign,
                                        SCALAR weight_rat_intermediate_state,
                                        const std::vector<psi> &creation_ops,
                                        const std::vector<psi> &annihilation_ops,
                                        const Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic> &M,
                                        boost::multi_array<std::complex<double>, 3> &result,
                                        const Eigen::Tensor<std::complex<double>,6>& trans_tensor_H_to_F
) {
  const double temperature = 1. / beta;
  const int num_flavors = result.shape()[0];
  const int dim_basis = p_basis_f->dim();

  std::vector<double> Pl_vals(dim_basis);

  std::vector<psi>::const_iterator it1, it2;
  const int mat_size = M.rows();

  assert(mat_size == creation_ops.size());

  //First, we compute relative weights
  boost::multi_array<SCALAR,2> coeffs(boost::extents[mat_size][mat_size]);
  double norm = 0.0;
  for (int k = 0; k < mat_size; k++) {//the last one is aux fields
    (k == 0 ? it1 = annihilation_ops.begin() : it1++);
    for (int l = 0; l < mat_size; l++) {
      (l == 0 ? it2 = creation_ops.begin() : it2++);
      if (M(l, k) == 0.0) {
        coeffs[k][l] = 0.0;
        continue;
      }

      const double bubble_sign = it1->time() - it2->time() > 0.0 ? 1.0 : -1.0;

      coeffs[k][l] = M(l, k) * bubble_sign * sign * weight_rat_intermediate_state;
      norm += std::abs(coeffs[k][l]);
    }
  }

  std::vector<double> sqrt_2l_1(dim_basis);
  for (int il = 0; il < dim_basis; ++il) {
    sqrt_2l_1[il] = sqrt(2.0/p_basis_f->norm2(il));
  }

  double scale_fact = -1.0/(norm * beta);
  for (int k = 0; k < mat_size; k++) {//the last one is aux fields
    (k == 0 ? it1 = annihilation_ops.begin() : it1++);
    for (int l = 0; l < mat_size; l++) {
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
      p_basis_f->value(x, Pl_vals);
      for (int il = 0; il < dim_basis; ++il) {
        result[flavor_a][flavor_c][il] += scale_fact * coeffs[k][l] * sqrt_2l_1[il] * Pl_vals[il];
      }
    }
  }
};


//Measure G2 by removal in G2 space
template<typename SCALAR>
void MeasureGHelper<SCALAR, 2>::perform(double beta,
                                        boost::shared_ptr<OrthogonalBasis> p_basis_f,
                                        boost::shared_ptr<OrthogonalBasis> p_basis_b,
                                        SCALAR sign,
                                        SCALAR weight_rat_intermediate_state,
                                        const std::vector<psi> &creation_ops,
                                        const std::vector<psi> &annihilation_ops,
                                        const Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic> &M,
                                        boost::multi_array<std::complex<double>, 7> &result,
                                        const Eigen::Tensor<std::complex<double>,6>& trans_tensor_H_to_F
) {
  const double temperature = 1. / beta;
  const int num_flavors = result.shape()[0];
  const int dim_f = p_basis_f->dim();
  const int dim_b = p_basis_b->dim();
  const int num_phys_rows = creation_ops.size();

  assert(M.rows() == creation_ops.size());

  if (creation_ops.size() != annihilation_ops.size() || creation_ops.size() != M.rows()) {
    throw std::runtime_error("Fatal error in MeasureGHelper<SCALAR, 2>::perform()");
  }

  boost::multi_array<double, 3>
      Pl_f(boost::extents[num_phys_rows][num_phys_rows][dim_f]);//annihilator, creator, ir
  boost::multi_array<double, 3>
      Pl_b(boost::extents[num_phys_rows][num_phys_rows][dim_b]);//annihilator, creator, ir
  {
    //Normalization factor of basis functions
    std::vector<double> norm_coeff_f(dim_f), norm_coeff_b(dim_b);
    for (int il = 0; il < dim_f; ++il) {
      norm_coeff_f[il] = sqrt(2.0/p_basis_f->norm2(il));
    }
    for (int il = 0; il < dim_b; ++il) {
      norm_coeff_b[il] = sqrt(2.0/p_basis_b->norm2(il));
    }

    std::vector<double> Pl_tmp(std::max(dim_f, dim_b));
    for (int k = 0; k < num_phys_rows; k++) {
      for (int l = 0; l < num_phys_rows; l++) {
        double argument = annihilation_ops[k].time() - creation_ops[l].time();
        double arg_sign = 1.0;
        if (argument < 0) {
          argument += beta;
          arg_sign = -1.0;
        }
        const double x = 2 * argument * temperature - 1.0;
        p_basis_f->value(x, Pl_tmp);
        for (int il = 0; il < dim_f; ++il) {
          Pl_f[k][l][il] = arg_sign * norm_coeff_f[il] * Pl_tmp[il];
        }

        p_basis_b->value(x, Pl_tmp);
        for (int il = 0; il < dim_b; ++il) {
          Pl_b[k][l][il] = norm_coeff_b[il] * Pl_tmp[il];
        }
      }
    }
  }

  //The indices of M are reverted from (C. 24) of L. Boehnke (2011) because we're using the F convention here.
  //First, compute relative weights. This costs O(num_phys_rows^4) operations.
  double norm = 0.0;
  for (int a = 0; a < num_phys_rows; ++a) {
    for (int b = 0; b < num_phys_rows; ++b) {
      for (int c = a+1; c < num_phys_rows; ++c) {
        for (int d = b+1; d < num_phys_rows; ++d) {
          /*
           * Delta convention
           * M_ab  M_ad
           * M_cb  M_cd
           */
          norm += std::abs(weight_rat_intermediate_state * (M(b,a) * M(d,c) - M(b,c) * M(d,a)));
        }
      }
    }
  }
  //The factor 4 is from the degree of freedom of exchange a and c or b and d.
  norm *= 4;

  Eigen::Tensor<SCALAR,3> tensor1(num_flavors, num_phys_rows, dim_f);//(f2, il1)
  tensor1.setZero();
  for (int il1 = 0; il1 < dim_f; ++il1) {
    for (int a = 0; a < num_phys_rows; ++a) {
      for (int b = 0; b < num_phys_rows; ++b) {
        int flavor_b = creation_ops[b].flavor();
        tensor1(flavor_b, a, il1) += M(b,a) * Pl_f[a][b][il1];
      }
    }
  }

  Eigen::Tensor<SCALAR,3> tensor2(num_flavors, num_phys_rows, dim_f);//(f3, d, il2)
  tensor2.setZero();
  for (int il2 = 0; il2 < dim_f; ++il2) {
    for (int d = 0; d < num_phys_rows; ++d) {
      for (int c = 0; c < num_phys_rows; ++c) {
        const int flavor_c = annihilation_ops[c].flavor();
        tensor2(flavor_c, d, il2) += M(d,c) * Pl_f[c][d][il2] * (il2%2 == 0 ? -1.0 : 1.0);
      }
    }
  }

  //Contraction requires O(num_phys_rows^2 Nl^3 num_flavors^2) operators
  //FIXME: USE CONTRACT IN EIGEN::TENSOR
  Eigen::Tensor<SCALAR,7> result_H(num_flavors, num_flavors, num_flavors, num_flavors, dim_f, dim_f, dim_b);
  result_H.setZero();
  const SCALAR coeff = sign * weight_rat_intermediate_state / (norm * beta);//where is this beta factor from?
  for (int a = 0; a < num_phys_rows; ++a) {
    const int flavor_a = annihilation_ops[a].flavor();
    for (int d = 0; d < num_phys_rows; ++d) {
      const int flavor_d = creation_ops[d].flavor();

      for (int il1 = 0; il1 < dim_f; ++il1) {
      for (int il2 = 0; il2 < dim_f; ++il2) {
      for (int il3 = 0; il3 < dim_b; ++il3) {
        for (int flavor_b = 0; flavor_b < num_flavors; ++flavor_b) {
        for (int flavor_c = 0; flavor_c < num_flavors; ++flavor_c) {
          result_H(flavor_a, flavor_b, flavor_c, flavor_d, il1, il2, il3) +=
            coeff * tensor1(flavor_b, a, il1) * tensor2(flavor_c, d, il2) * Pl_b[a][d][il3];
        }
        }
      }
      }
      }
    }
  }

  //substract contributions from terms for a==c or b==d.
  /*
  for (int a = 0; a < num_phys_rows; ++a) {
    int flavor_a = annihilation_ops[a].flavor();
    for (int b = 0; b < num_phys_rows; ++b) {
      int flavor_b = creation_ops[b].flavor();
      for (int c = 0; c < num_phys_rows; ++c) {
        int flavor_c = annihilation_ops[c].flavor();
        for (int d = 0; d < num_phys_rows; ++d) {
          int flavor_d = creation_ops[d].flavor();

          if (a != c && b != d) {
            continue;
          }

          for (int il1 = 0; il1 < dim_f; ++il1) {
            for (int il2 = 0; il2 < dim_f; ++il2) {
              for (int il3 = 0; il3 < dim_b; ++il3) {
                result_H(flavor_a, flavor_b, flavor_c, flavor_d, il1, il2, il3) -=
                    coeff * M(b, a) * M(d, c) * Pl_f[a][b][il1] * Pl_f[c][d][il2] * Pl_b[a][d][il3]
                        * (il2%2 == 0 ? -1.0 : 1.0);
              }
            }
          }
        }
      }
    }
  }
  */

  //DEBUG
//#define COMPUTE_F

#ifdef COMPUTE_F
  Eigen::Tensor<SCALAR,7> result_F(num_flavors, num_flavors, num_flavors, num_flavors, dim_f, dim_f, dim_b);
  result_F.setZero();
  Eigen::Tensor<SCALAR,7> result_H2(num_flavors, num_flavors, num_flavors, num_flavors, dim_f, dim_f, dim_b);
  result_H2.setZero();
  for (int a = 0; a < num_phys_rows; ++a) {
    int flavor_a = annihilation_ops[a].flavor();
    for (int b = 0; b < num_phys_rows; ++b) {
      int flavor_b = creation_ops[b].flavor();
      for (int c = 0; c < num_phys_rows; ++c) {
        int flavor_c = annihilation_ops[c].flavor();
        for (int d = 0; d < num_phys_rows; ++d) {
          int flavor_d = creation_ops[d].flavor();

          if (a == c || b == d) {
            continue;
          }

          for (int il1 = 0; il1 < dim_f; ++il1) {
            for (int il2 = 0; il2 < dim_f; ++il2) {
              for (int il3 = 0; il3 < dim_b; ++il3) {
                result_F(flavor_a, flavor_b, flavor_c, flavor_d, il1, il2, il3) +=
                    -coeff * M(d, a) * M(b, c) * Pl_f[a][b][il1] * Pl_f[c][d][il2] * Pl_b[a][d][il3]
                        * (il2%2 == 0 ? -1.0 : 1.0);
                result_H2(flavor_a, flavor_b, flavor_c, flavor_d, il1, il2, il3) +=
                    coeff * M(b, a) * M(d, c) * Pl_f[a][b][il1] * Pl_f[c][d][il2] * Pl_b[a][d][il3]
                        * (il2%2 == 0 ? -1.0 : 1.0);
              }
            }
          }
        }
      }
    }
  }
#endif

  //Then, accumulate data
  for (int flavor_a = 0; flavor_a < num_flavors; ++flavor_a) {
  for (int flavor_b = 0; flavor_b < num_flavors; ++flavor_b) {
  for (int flavor_c = 0; flavor_c < num_flavors; ++flavor_c) {
  for (int flavor_d = 0; flavor_d < num_flavors; ++flavor_d) {
    for (int il1 = 0; il1 < dim_f; ++il1) {
    for (int il2 = 0; il2 < dim_f; ++il2) {
    for (int il3 = 0; il3 < dim_b; ++il3) {
      //if(std::abs(result_H(flavor_a, flavor_b, flavor_c, flavor_d, il1, il2, il3)-result_H2(flavor_a, flavor_b, flavor_c, flavor_d, il1, il2, il3))>1e-5) {
        //throw std::runtime_error("Error in H and H2.");
      //}
#ifdef COMPUTE_F
      result[flavor_a][flavor_b][flavor_c][flavor_d][il1][il2][il3] +=
          //result_F(flavor_a, flavor_b, flavor_c, flavor_d, il1, il2, il3) + result_H(flavor_a, flavor_b, flavor_c, flavor_d, il1, il2, il3);
          result_H(flavor_a, flavor_b, flavor_c, flavor_d, il1, il2, il3);
#else
      result[flavor_a][flavor_b][flavor_c][flavor_d][il1][il2][il3] += result_H(flavor_a, flavor_b, flavor_c, flavor_d, il1, il2, il3);
#endif
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
