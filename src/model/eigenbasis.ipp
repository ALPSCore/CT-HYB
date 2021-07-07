template<typename SCALAR>
AtomicModelEigenBasis<SCALAR>::AtomicModelEigenBasis(const alps::params &par, bool verbose)
    : AtomicModel<SCALAR, AtomicModelEigenBasis<SCALAR> >(par, verbose) {
  build_basis(par["inner_outer_cutoff_energy"].template as<double>());
  build_outer_braket(par["outer_cutoff_energy"].template as<double>());
  build_qops();
}

template<typename SCALAR>
AtomicModelEigenBasis<SCALAR>::AtomicModelEigenBasis(int nflavors,
                                                         const std::vector<std::tuple<int, int, SCALAR> > &nonzero_t_vals_list,
                                                         const std::vector<std::tuple<int, int, int, int, SCALAR> > &nonzero_U_vals_list,
                                                         bool verbose,
                                                         double cutoff_ham,
                                                         double hermicity_tolerance,
                                                         double inner_outer_cutoff_energy,
                                                         double outer_cutoff_energy
                                                         )
    : AtomicModel<SCALAR, AtomicModelEigenBasis<SCALAR> >(nflavors, nonzero_t_vals_list, nonzero_U_vals_list, verbose) {
  build_basis(inner_outer_cutoff_energy);
  build_outer_braket(outer_cutoff_energy);
  build_qops();
}

template<typename SCALAR>
void AtomicModelEigenBasis<SCALAR>::define_parameters(alps::params &parameters) {
  Base::define_parameters(parameters);
  parameters
    .define<double>("model.inner_outer_cutoff_energy", 0.1 * std::numeric_limits<double>::max(),
      "Cutoff energy for inner states for computing trace (measured from the lowest eigenvalue)")
    .define<double>("model.outer_cutoff_energy", 0.1 * std::numeric_limits<double>::max(),
                      "Cutoff energy for outer states for computing trace (measured from the lowest eigenvalue)");
}

template<typename SCALAR, typename OP>
void construct_operator_object(const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> &mat, OP &op_obj) {
  op_obj.resize(mat.rows(), mat.cols());
  for (int j = 0; j < mat.cols(); ++j) {
    for (int i = 0; i < mat.rows(); ++i) {
      op_obj(i, j) = mat(i, j);
    }
  }
};

inline void print_sectors(const std::vector<std::vector<double> > &evals_sectors) {
  const int num_sectors = evals_sectors.size();
  for (int sector = 0; sector < num_sectors; ++sector) {
    if (evals_sectors[sector].size() == 0) {
      continue;
    }
    std::cout << "Sector " << sector << " : dim = " << evals_sectors[sector].size()
        << ", min energy = "
        << *std::min_element(evals_sectors[sector].begin(), evals_sectors[sector].end())
        << ", max energy = "
        << *std::max_element(evals_sectors[sector].begin(), evals_sectors[sector].end()) << std::endl;
  }
}

template<typename M>
void remove_high_energy_states(std::vector<std::vector<double> > &evals_sectors,
                               std::vector<M> &evecs_sectors,
                               double cutoff_energy) {
  namespace bll = boost::lambda;

  const int num_sectors = evals_sectors.size();
  for (int sector = 0; sector < num_sectors; ++sector) {
    const int num_e = evals_sectors[sector].size();
    const int num_e_active = std::count_if(evals_sectors[sector].begin(),
                                           evals_sectors[sector].end(),
                                           bll::_1 <= cutoff_energy);
    if (num_e_active == 0) {
      evals_sectors[sector].resize(0);
      evecs_sectors[sector].resize(0, 0);
      continue;
    }

    const int dim_org = size1(evecs_sectors[sector]);
    M evecs_new(dim_org, num_e_active);
    std::vector<double> evals_new(num_e_active);

    int ie_active = 0;
    for (int ie = 0; ie < num_e; ++ie) {
      if (evals_sectors[sector][ie] > cutoff_energy) {
        continue;
      }
      evals_new[ie_active] = evals_sectors[sector][ie];
      evecs_new.col(ie_active) = evecs_sectors[sector].col(ie);
      ++ie_active;
    }
    std::swap(evals_sectors[sector], evals_new);
    std::swap(evecs_sectors[sector], evecs_new);
    assert(ie_active == num_e_active);
  }
  assert(evals_sectors.size() == num_sectors);
}

inline std::pair<double, double>
min_max_energy_sectors(const std::vector<std::vector<double> > &evals_sectors,
                       std::vector<double> &min_eigenval_sector) {
  const int num_sectors = evals_sectors.size();
  double eigenvalue_max = -std::numeric_limits<double>::max();
  double eigenvalue_min = std::numeric_limits<double>::max();
  for (int sector = 0; sector < num_sectors; ++sector) {
    if (evals_sectors[sector].size() == 0) {
      continue;
    }
    const double min_tmp = *std::min_element(evals_sectors[sector].begin(), evals_sectors[sector].end());
    const double max_tmp = *std::max_element(evals_sectors[sector].begin(), evals_sectors[sector].end());
    min_eigenval_sector[sector] = min_tmp;
    eigenvalue_max = std::max(eigenvalue_max, max_tmp);
    eigenvalue_min = std::min(eigenvalue_min, min_tmp);
  }
  return std::make_pair(eigenvalue_max, eigenvalue_min);
}

template<typename SCALAR>
bool AtomicModelEigenBasis<SCALAR>::is_sector_active(int sector) const {
  if (sector == nirvana || eigenvals_sector[sector].size() == 0) {
    return false;
  } else {
    return true;
  }
}

template<typename SCALAR>
void AtomicModelEigenBasis<SCALAR>::build_basis(double inner_outer_cutoff_energy) {
  //build eigenbasis
  const int num_sectors = Base::num_sectors();
  const std::vector<std::vector<int> > &sector_members = Base::get_sector_members();
  const int flavors = Base::num_flavors();
  typedef Eigen::SelfAdjointEigenSolver<dense_matrix_t> SOLVER_TYPE;
  std::vector<dense_matrix_t> ham_sector;

  //Compute eigenvectors and eigenvalues
  min_eigenval_sector.resize(num_sectors);
  eigenvals_sector.resize(num_sectors);
  std::vector<dense_matrix_t> evecs_sector(num_sectors);
  for (int sector = 0; sector < num_sectors; ++sector) {
    const int dim_sector = sector_members[sector].size();
    dense_matrix_t ham_tmp(Base::ham_sectors[sector]);
    ham_sector.push_back(ham_tmp);
    SOLVER_TYPE esolv(ham_tmp);
    eigenvals_sector[sector].resize(dim_sector);
    for (int ie = 0; ie < dim_sector; ++ie) {
      eigenvals_sector[sector][ie] = esolv.eigenvalues()[ie];
      evecs_sector[sector] = esolv.eigenvectors();
    }
  }

  //Compute the lowest eigenenergy
  double eigenvalue_max, eigenvalue_min;
  boost::tie(eigenvalue_max, eigenvalue_min) = min_max_energy_sectors(eigenvals_sector, min_eigenval_sector);
  if (Base::verbose_) {
    print_sectors(eigenvals_sector);
    std::cout << " Max eigen energy = " << eigenvalue_max << std::endl;
    std::cout << " Min eigen energy = " << eigenvalue_min << std::endl;
  }

  //Throwing away high-energy states
  if (Base::verbose_) {
    std::cout << " Throwing away high energy states..." << std::endl;
  }
  remove_high_energy_states(eigenvals_sector,
                            evecs_sector,
                            eigenvalue_min + inner_outer_cutoff_energy);
  boost::tie(eigenvalue_max, eigenvalue_min) = min_max_energy_sectors(eigenvals_sector, min_eigenval_sector);
  if (Base::verbose_) {
    print_sectors(eigenvals_sector);
    std::cout << " Max eigen energy = " << eigenvalue_max << std::endl;
    std::cout << " Min eigen energy  = " << eigenvalue_min << std::endl;
  }
  check_evecs(ham_sector, evecs_sector);

  //modify sector_connection
  for (int op = 0; op < 2; ++op) {
    for (int flavor = 0; flavor < flavors; ++flavor) {
      for (int src_sector = 0; src_sector < num_sectors; ++src_sector) {
        if (!is_sector_active(Base::sector_connection[op][flavor][src_sector])) {
          Base::sector_connection[op][flavor][src_sector] = nirvana;
        }
        if (!is_sector_active(Base::sector_connection_reverse[op][flavor][src_sector])) {
          Base::sector_connection_reverse[op][flavor][src_sector] = nirvana;
        }
      }
    }
  }

  //overflow prevention
  Base::reference_energy_ = eigenvalue_min;
  /*
  double overflow_prevention;
  const double BETA = par["model.beta"].template as<double>();
  if (0.5 * (eigenvalue_max - eigenvalue_min) > 200 / BETA) {
    overflow_prevention = 0.5 * (eigenvalue_max - eigenvalue_min) - 200 / BETA;
  } else {
    overflow_prevention = 0;
  }
  Base::reference_energy_ = eigenvalue_min;
  */
  if (Base::verbose_) {
    std::cout << "Reference energy " << Base::reference_energy_ << std::endl;
  }
  for (int sector = 0; sector < num_sectors; ++sector) {
    for (int ie = 0; ie < eigenvals_sector[sector].size(); ++ie) {
      eigenvals_sector[sector][ie] -= Base::reference_energy_;
    }
    min_eigenval_sector[sector] -= Base::reference_energy_;
  }

  //transform d, d^dagger to eigenbasis
  ddag_ops_eigen.resize(flavors);
  d_ops_eigen.resize(flavors);
  for (int flavor = 0; flavor < flavors; ++flavor) {
    ddag_ops_eigen[flavor].resize(num_sectors);
    d_ops_eigen[flavor].resize(num_sectors);
    for (int src_sector = 0; src_sector < num_sectors; ++src_sector) {
      int dst_sector = Base::get_dst_sector_ket(CREATION_OP, flavor, src_sector);
      if (!is_sector_active(dst_sector) || !is_sector_active(src_sector)) {
        ddag_ops_eigen[flavor][src_sector].resize(0, 0);
      } else {
        dense_matrix_t tmp_mat = evecs_sector[dst_sector].adjoint() * Base::creation_operators_hyb(flavor, src_sector)
            * evecs_sector[src_sector];
        construct_operator_object(tmp_mat, ddag_ops_eigen[flavor][src_sector]);
      }

      dst_sector = Base::get_dst_sector_ket(ANNIHILATION_OP, flavor, src_sector);
      if (!is_sector_active(dst_sector) || !is_sector_active(src_sector)) {
        d_ops_eigen[flavor][src_sector].resize(0, 0);
      } else {
        dense_matrix_t tmp_mat =
            evecs_sector[dst_sector].adjoint() * Base::annihilation_operators_hyb(flavor, src_sector)
                * evecs_sector[src_sector];
        construct_operator_object(tmp_mat, d_ops_eigen[flavor][src_sector]);
      }
    }
  }
}

// Build matrix representations of q and q^dagger operator in eigen basis
// q_a = - [H_loc, d_a] = d_a H_loc - H_loc d_a
// q^dagger_a = [H_loc, d^dagger_a] = H_loc d^dagger_a - d^dagger_a H_loc
template<typename SCALAR>
void AtomicModelEigenBasis<SCALAR>::build_qops() {
  q_ops_eigen.resize(this->num_flavors());
  qdag_ops_eigen.resize(this->num_flavors());
  for (auto f=0; f<this->num_flavors(); ++f) {
    q_ops_eigen[f].resize(this->num_sectors());
    qdag_ops_eigen[f].resize(this->num_flavors());

    // Build q ops
    for (auto sector=0; sector<this->num_sectors(); ++sector) {
      if (d_ops_eigen[f][sector].rows() == 0 || d_ops_eigen[f][sector].cols() == 0) {
        continue;
      }
      dense_matrix_t q1(d_ops_eigen[f][sector]); //Hloc d
      dense_matrix_t q2(d_ops_eigen[f][sector]); //d Hloc
      for (auto j=0; j<q1.cols(); ++j) {
        q1.col(j) *= eigenvals_sector[sector][j];
      }
      for (auto i=0; i<q2.rows(); ++i) {
        q2.row(i) *= eigenvals_sector[sector][i];
      }
      q_ops_eigen[f][sector] = q1 - q2;
    }

    // Build qdagger ops
    for (auto sector=0; sector<this->num_sectors(); ++sector) {
      if (ddag_ops_eigen[f][sector].rows() == 0 || ddag_ops_eigen[f][sector].cols() == 0) {
        continue;
      }
      dense_matrix_t qdagg1(ddag_ops_eigen[f][sector]); //Hloc d_dagg
      dense_matrix_t qdagg2(ddag_ops_eigen[f][sector]); //d_dagg Hloc
      for (auto j=0; j<qdagg1.cols(); ++j) {
        qdagg1.col(j) *= eigenvals_sector[sector][j];
      }
      for (auto i=0; i<qdagg2.rows(); ++i) {
        qdagg2.row(i) *= eigenvals_sector[sector][i];
      }
      q_ops_eigen[f][sector] = - qdagg1 + qdagg2;
    }
  }
}

template<typename SCALAR>
void AtomicModelEigenBasis<SCALAR>::build_outer_braket(double outer_cutoff_energy) {
  namespace bll = boost::lambda;
  const double cutoff_outer = outer_cutoff_energy + 
    *std::min_element(min_eigenval_sector.begin(), min_eigenval_sector.end());
  int active_sector = 0;
  bra_list.resize(0);
  ket_list.resize(0);
  for (int sector = 0; sector < Base::num_sectors(); ++sector) {
    if (!is_sector_active(sector)) {
      continue;
    }

    const int dim = dim_sector(sector);
    const int dim_outer =
        std::count_if(eigenvals_sector[sector].begin(), eigenvals_sector[sector].end(), bll::_1 <= cutoff_outer);
    if (dim_outer == 0) {
      continue;
    }

    braket_obj_t obj;
    obj.resize(dim, dim_outer);
    obj.setZero();

    int active_outer = 0;
    for (int outer = 0; outer < dim; ++outer) {
      if (eigenvals_sector[sector][outer] <= cutoff_outer) {
        obj(outer, active_outer) = 1.0;
        ++active_outer;
      }
    }
    assert(active_outer == dim_outer);

    bra_list.push_back(BRAKET_T(sector, obj.transpose()));
    ket_list.push_back(BRAKET_T(sector, obj));

    if (Base::verbose_) {
      std::cout << "Dim of ket: sector " << sector << " inner " << dim << " outer " << dim_outer << std::endl;
    }
    ++active_sector;
  }
  num_braket_ = active_sector;
}

template<typename SCALAR>
void AtomicModelEigenBasis<SCALAR>::check_evecs(const std::vector<dense_matrix_t> ham_sector,
                                                  const std::vector<dense_matrix_t> &evecs_sector) {
#ifndef NDEBUG
  const int num_sectors = evecs_sector.size();

  for (int sector = 0; sector < num_sectors; ++sector) {
    for (int ie = 0; ie < eigenvals_sector[sector].size(); ++ie) {
      const double res_norm = (ham_sector[sector] * evecs_sector[sector].col(ie)
          - eigenvals_sector[sector][ie] * evecs_sector[sector].col(ie)).squaredNorm();
      assert(res_norm / eigenvals_sector[sector].size() < 1E-8);
    }
  }
#endif
}

template<typename SCALAR>
void AtomicModelEigenBasis<SCALAR>::apply_op_ket_impl(
  const OPERATOR_TYPE &op_type, int flavor, BRAKET_T &ket,
  const std::vector<std::vector<dense_matrix_t>> &ops) const {
  using namespace std;

  if (ket.invalid()) {
    return;
  }

  const int sector_new = Base::get_dst_sector_ket(op_type, flavor, ket.sector());
  if (sector_new == nirvana) {
    ket.set_invalid();
    return;
  }

  EXTENDED_REAL max_norm_old = ket.max_norm();
  dense_matrix_t work_mat = ops[flavor][ket.sector()] * ket.obj();
  ket.swap_obj(work_mat);
  ket.set_sector(sector_new);

  if (ket.max_norm() / max_norm_old < 1E-30) {
    ket.set_invalid();
  }
}


template<typename SCALAR>
void AtomicModelEigenBasis<SCALAR>::apply_op_bra_impl(
  const OPERATOR_TYPE &op_type, int flavor, BRAKET_T &bra,
  const std::vector<std::vector<dense_matrix_t>> &ops) const {
  using std::swap;

  if (bra.invalid()) {
    return;
  }

  const int sector_new = Base::get_dst_sector_bra(op_type, flavor, bra.sector());
  if (sector_new == nirvana) {
    bra.set_invalid();
    return;
  }

  EXTENDED_REAL max_norm_old = bra.max_norm();

  dense_matrix_t work_mat = bra.obj() * ops[flavor][sector_new];
  bra.swap_obj(work_mat);
  bra.set_sector(sector_new);

  if (bra.max_norm() / max_norm_old < 1E-30) {
    bra.set_invalid();
  }
}

template<typename SCALAR>
void AtomicModelEigenBasis<SCALAR>::apply_op_hyb_ket(
    const OPERATOR_TYPE &op_type, int flavor, BRAKET_T &ket) const {
  check_true(op_type == CREATION_OP || op_type == ANNIHILATION_OP);
  if (op_type == CREATION_OP) {
    apply_op_ket_impl(op_type, flavor, ket, ddag_ops_eigen);
  } else {
    apply_op_ket_impl(op_type, flavor, ket, d_ops_eigen);
  }
}

template<typename SCALAR>
void AtomicModelEigenBasis<SCALAR>::apply_op_hyb_bra(
  const OPERATOR_TYPE &op_type, int flavor, BRAKET_T &bra) const {
  check_true(op_type == CREATION_OP || op_type == ANNIHILATION_OP);
  if (op_type == CREATION_OP) {
    apply_op_bra_impl(op_type, flavor, bra, ddag_ops_eigen);
  } else {
    apply_op_bra_impl(op_type, flavor, bra, d_ops_eigen);
  }
}

template<typename SCALAR>
void AtomicModelEigenBasis<SCALAR>::apply_qop_ket(
    const OPERATOR_TYPE &op_type, int flavor, BRAKET_T &ket) const {
  check_true(op_type == CREATION_OP || op_type == ANNIHILATION_OP);
  if (op_type == CREATION_OP) {
    apply_op_ket_impl(op_type, flavor, ket, qdag_ops_eigen);
  } else {
    apply_op_ket_impl(op_type, flavor, ket, q_ops_eigen);
  }
}

template<typename SCALAR>
void AtomicModelEigenBasis<SCALAR>::apply_qop_bra(
  const OPERATOR_TYPE &op_type, int flavor, BRAKET_T &bra) const {
  check_true(op_type == CREATION_OP || op_type == ANNIHILATION_OP);
  if (op_type == CREATION_OP) {
    apply_op_bra_impl(op_type, flavor, bra, qdag_ops_eigen);
  } else {
    apply_op_bra_impl(op_type, flavor, bra, q_ops_eigen);
  }
}

template<typename SCALAR>
typename ExtendedScalar<SCALAR>::value_type
AtomicModelEigenBasis<SCALAR>::product(const BRAKET_T &bra, const BRAKET_T &ket) const {
  if (bra.invalid() || ket.invalid() || bra.sector() != ket.sector()) {
    return 0.0;
  }
  if (size2(bra.obj()) != size1(ket.obj())) {
    throw std::runtime_error("Something got wrong in product!");
  }
  if (size1(bra.obj()) != size2(ket.obj())) {
    throw std::runtime_error("Something got wrong in product!");
  }
  return static_cast<typename ExtendedScalar<SCALAR>::value_type>(bra.coeff() * ket.coeff()) * (bra.obj() * ket.obj()).trace();
}

template<typename SCALAR>
void AtomicModelEigenBasis<SCALAR>::sector_propagate_ket(BRAKET_T &ket, double t) const {
  if (ket.invalid()) {
    return;
  }

  const int dim = dim_sector(ket.sector());
  const int rows = size1(ket.obj());
  const int cols = size2(ket.obj());
  assert(rows == dim);

  std::vector<double> exp_v(dim);
  const int sector = ket.sector();
  assert(eigenvals_sector.size() > sector);

  const double coeff = compute_exp_vector_safe(t, eigenvals_sector[sector], exp_v);

  //Assuming the matrix in clumn major format
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      ket.obj()(i, j) *= exp_v[i];
    }
  }
  ket.set_coeff(ket.coeff() * coeff);
}

template<typename SCALAR>
void AtomicModelEigenBasis<SCALAR>::sector_propagate_bra(BRAKET_T &bra, double t) const {
  if (bra.invalid()) {
    return;
  }

  const int sector = bra.sector();
  const int dim = dim_sector(sector);
  std::vector<double> exp_v(dim);
  const double coeff = compute_exp_vector_safe(t, eigenvals_sector[sector], exp_v);

  const int rows = size1(bra.obj());
  const int cols = size2(bra.obj());
  assert(cols == dim);

  //Assuming the matrix in clumn major format
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      bra.obj()(i, j) *= exp_v[j];
    }
  }
  bra.set_coeff(bra.coeff() * coeff);
}

template<typename SCALAR>
typename model_traits<AtomicModelEigenBasis<SCALAR> >::BRAKET_T AtomicModelEigenBasis<SCALAR>::get_outer_bra(int bra) const {
  assert(bra >= 0 && bra < num_brakets());
  return bra_list[bra];
}

template<typename SCALAR>
typename model_traits<AtomicModelEigenBasis<SCALAR> >::BRAKET_T AtomicModelEigenBasis<SCALAR>::get_outer_ket(int ket) const {
  assert(ket >= 0 && ket < num_brakets());
  return ket_list[ket];
}

template<typename SCALAR>
bool AtomicModelEigenBasis<SCALAR>::translationally_invariant() const {
  namespace bll = boost::lambda;

  int tot_dim = 0;
  for (int bra = 0; bra < num_braket_; ++bra) {
    tot_dim += std::min(size1(bra_list[bra].obj()), size2(bra_list[bra].obj()));
  }
  return std::abs(tot_dim - std::pow(2.0, Base::num_flavors())) < 1E-5;
}
