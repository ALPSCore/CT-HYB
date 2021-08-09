#include "../hdf5/boost_any.hpp"

template<typename SCALAR, typename DERIVED>
AtomicModel<SCALAR, DERIVED>::AtomicModel(const alps::params &par, bool verbose)
    : flavors_(par["model.sites"].template as<int>() * par["model.spins"].template as<int>()),
      dim_(1 << flavors_),
      reference_energy_(-1E+100),//this should be set in a derived class,
      verbose_(verbose),
      hopping_mat(flavors_, flavors_),
      U_tensor(flavors_, flavors_, flavors_, flavors_)
      {
  read_U_tensor(par["model.coulomb_tensor_input_file"].template as<std::string>(), flavors_, nonzero_U_vals);
  read_hopping(par["model.hopping_matrix_input_file"].template as<std::string>(), flavors_, nonzero_t_vals);
  hilbert_space_partioning(
      par["model.cutoff_ham"].template as<double>(),
      par["model.hermicity_tolerance"].template as<double>()
  );
  init_nelec_sectors();
}

template<typename SCALAR, typename DERIVED>
void AtomicModel<SCALAR, DERIVED>::define_parameters(alps::params &parameters) {
  parameters
      .define<std::string>("model.coulomb_tensor_input_file", "Input file containing nonzero elements of U tensor")
      .define<std::string>("model.hopping_matrix_input_file", "Input file for hopping matrix")
      .define<std::string>("model.delta_input_file", "", "Input file for hybridization function Delta(tau)")
      .define<double>("model.cutoff_ham", 1E-12,
                      "Cutoff for entries in the local Hamiltonian matrix")
      .define<double>("model.hermicity_tolerance", 1E-12,
                      "Tolerance in checking hermicity of the local Hamiltonian matrix");
}

template<typename SCALAR, typename DERIVED>
void AtomicModel<SCALAR, DERIVED>::save_info_for_postprocessing(const std::string &filename) const {
  alps::hdf5::archive oar(filename, "a");
  oar["/hopping"] << boost::any(hopping_mat);
  oar["/U_tensor"] << boost::any(U_tensor);
}

//mainly for unitest
template<typename SCALAR, typename DERIVED>
AtomicModel<SCALAR, DERIVED>::AtomicModel(int nflavors,
                                              const std::vector<std::tuple<int, int, SCALAR> > &nonzero_t_vals_list,
                                              const std::vector<std::tuple<int, int, int, int, SCALAR> > &nonzero_U_vals_list,
                                              bool verbose,
                                              double cutoff_ham,
                                              double hermicity_tolerance
                                              )
    : flavors_(nflavors),
      dim_(1 << flavors_),
      verbose_(verbose),
      nonzero_U_vals(nonzero_U_vals_list),
      nonzero_t_vals(nonzero_t_vals_list),
      hopping_mat(flavors_, flavors_),
      U_tensor(flavors_, flavors_, flavors_, flavors_) {
  hilbert_space_partioning(cutoff_ham, hermicity_tolerance);
  init_nelec_sectors();
}


template<typename SCALAR, typename DERIVED>
AtomicModel<SCALAR, DERIVED>::~AtomicModel() { }

template<typename SCALAR, typename DERIVED>
int AtomicModel<SCALAR, DERIVED>::get_dst_sector_ket(OPERATOR_TYPE op, int flavor, int src_sector) const {
  return sector_connection[static_cast<int>(op)][flavor][src_sector];
}

template<typename SCALAR, typename DERIVED>
int AtomicModel<SCALAR, DERIVED>::get_dst_sector_bra(OPERATOR_TYPE op, int flavor, int src_sector) const {
  return sector_connection_reverse[static_cast<int>(op)][flavor][src_sector];
}

template<typename M, typename M2, typename P>
void merge_according_to_c_or_cdag(const M &mat, M2 &block_mat, const P &p, P &p2) {
  std::vector<int> rows;

  const int n_c = p.get_num_clusters();
  block_mat.resize(n_c, n_c);
  block_mat.setZero();

  const std::vector<int> &c_labels = p.get_cluster_labels();

  for (int k = 0; k < mat.outerSize(); ++k) {
    for (typename M::InnerIterator it(mat, k); it; ++it) {
      block_mat(c_labels[it.row()], c_labels[it.col()]) += 1.0;
    }
  }

  //search along each row
  for (int i = 0; i < n_c; ++i) {
    rows.resize(0);
    for (int j = 0; j < n_c; ++j) {
      if (block_mat(i, j) != 0.0) {
        rows.push_back(j);
      }
    }
    for (int row_idx = 1; row_idx < rows.size(); ++row_idx) {
      p2.connect_vertices(rows[0], rows[row_idx]);
    }
  }
}

template<typename T, typename IT>
void
split_op_into_sectors(int num_sectors,
                      const typename Eigen::SparseMatrix<T> &op,
                      const std::vector<int> &dim_sectors,
                      const std::vector<int> &index_of_state_in_sector,
                      const std::vector<int> &sector_of_state,
                      IT p_dst_sectors,
                      std::vector<Eigen::SparseMatrix<T> > &op_sectors,
                      bool assume_diagonal_block_matrix = false

) {
  typedef Eigen::SparseMatrix<T> sparse_matrix_t;
  typedef Eigen::Triplet<T> Tr;
  std::vector<std::vector<Tr> > triplets(num_sectors);

  std::fill(p_dst_sectors, p_dst_sectors + num_sectors, -1);
  op_sectors.resize(num_sectors);

  for (int k = 0; k < op.outerSize(); ++k) {
    for (typename sparse_matrix_t::InnerIterator it(op, k); it; ++it) {
      const int src_sector = sector_of_state[it.col()];
      *(p_dst_sectors + src_sector) = sector_of_state[it.row()];
      const int row_in_sector = index_of_state_in_sector[it.row()];
      const int col_in_sector = index_of_state_in_sector[it.col()];
      triplets[src_sector].push_back(Tr(row_in_sector, col_in_sector, it.value()));
    }
  }
  if (assume_diagonal_block_matrix) {
    for (int src_sector = 0; src_sector < num_sectors; ++src_sector) {
      *(p_dst_sectors + src_sector) = src_sector;
    }
  }
  for (int src_sector = 0; src_sector < num_sectors; ++src_sector) {
    if (*(p_dst_sectors + src_sector) < 0) {
      continue;
    }
    op_sectors[src_sector].resize(dim_sectors[*(p_dst_sectors + src_sector)], dim_sectors[src_sector]);
    op_sectors[src_sector].setFromTriplets(triplets[src_sector].begin(), triplets[src_sector].end());
  }
}


template<typename SCALAR, typename DERIVED>
void AtomicModel<SCALAR, DERIVED>::hilbert_space_partioning(double cutoff_ham, double hermicity_tolerance){
  const double eps_numerics = 1E-12;
  const double eps = cutoff_ham;

  // Construct hopping matrix
  hopping_mat.set_zero();
  for (int elem = 0; elem < nonzero_t_vals.size(); ++elem) {
    const int i = std::get<0>(nonzero_t_vals[elem]);
    const int j = std::get<1>(nonzero_t_vals[elem]);
    const SCALAR hopping = std::get<2>(nonzero_t_vals[elem]);
    hopping_mat(i, j) = hopping;
  }
  for (int flavor = 0; flavor < flavors_; ++flavor) {
    for (int flavor2 = 0; flavor2 < flavors_; ++flavor2) {
      if (std::abs(hopping_mat(flavor, flavor2) - myconj<SCALAR>(hopping_mat(flavor2, flavor))) > hermicity_tolerance) {
        throw std::runtime_error("Error: Hopping matrix is not hermite!");
      }
    }
  }
  for (int flavor2 = 0; flavor2 < flavors_; ++flavor2) {
    for (int flavor1 = 0; flavor1 < flavors_; ++flavor1) {
      if (std::abs(hopping_mat(flavor1, flavor2)) < eps_numerics) {
        hopping_mat(flavor1, flavor2) = 0;
      }
    }
  }

  // Construct U_tensor
  U_tensor.set_zero();
  for (int elem = 0; elem < nonzero_U_vals.size(); ++elem) {
    auto uval = std::get<4>(nonzero_U_vals[elem]);
    if (std::abs(uval) > eps_numerics) {
      U_tensor
        (std::get<0>(nonzero_U_vals[elem]),
         std::get<1>(nonzero_U_vals[elem]),
         std::get<2>(nonzero_U_vals[elem]),
         std::get<3>(nonzero_U_vals[elem])) += 2.0*uval;
    }
  }

  //Build sparse matrix representation of fermionic operators
  std::vector<sparse_matrix_t> d_ops, ddag_ops;
  {
    FermionOperator<SCALAR> fermion_op(flavors_);
    for (int flavor1 = 0; flavor1 < flavors_; ++flavor1) {
      d_ops.push_back(fermion_op.get_c(flavor1));
      ddag_ops.push_back(fermion_op.get_cdag(flavor1));
    }
  }

  //Build sparse matrix representation of Hamiltonian
  sparse_matrix_t ham(dim_, dim_);
  for (int elem = 0; elem < nonzero_U_vals.size(); ++elem) {
    auto uval = std::get<4>(nonzero_U_vals[elem]);
    if (std::abs(uval) > eps_numerics) {
      ham += uval
        * ddag_ops[std::get<0>(nonzero_U_vals[elem])]
        * ddag_ops[std::get<1>(nonzero_U_vals[elem])]
        * d_ops[   std::get<2>(nonzero_U_vals[elem])]
        * d_ops[   std::get<3>(nonzero_U_vals[elem])];
    }
  }

  for (int flavor2 = 0; flavor2 < flavors_; ++flavor2) {
    for (int flavor1 = 0; flavor1 < flavors_; ++flavor1) {
      ham += hopping_mat(flavor1, flavor2) * ddag_ops[flavor1] * d_ops[flavor2];
    }
  }
  ham.prune(PruneHelper<SCALAR>(eps));

  //Partionining of Hilbert space according to block structure of the local Hamiltonian
  Clustering cl(dim_);
  for (int k = 0; k < ham.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<SCALAR>::InnerIterator it(ham, k); it; ++it) {
      cl.connect_vertices(it.row(), it.col());
    }
  }
  cl.finalize_labeling();

  if (verbose_) {
    logger_out << "dim of Hilbert space " << dim_ << std::endl;
    logger_out << "# of blocks " << cl.get_num_clusters() << std::endl;
  }

  //Merge some blocks according to creation and annihilation operators
  Clustering cl2(cl.get_num_clusters());
  const int num_c = cl.get_num_clusters();
  real_matrix_t block_mat(num_c, num_c);
  for (int flavor = 0; flavor < flavors_; ++flavor) {
    merge_according_to_c_or_cdag(ddag_ops[flavor], block_mat, cl, cl2);
    merge_according_to_c_or_cdag(d_ops[flavor], block_mat, cl, cl2);
  }
  cl2.finalize_labeling();

  if (verbose_) {
    logger_out << "# of sectors " << cl2.get_num_clusters() << std::endl;
  }

  //identity members in each symmetry sector
  num_sectors_ = cl2.get_num_clusters();
  sector_members.resize(num_sectors_);
  sector_of_state.resize(dim_);
  index_of_state_in_sector.resize(dim_);
  for (int state = 0; state < dim_; ++state) {
    int sector_tmp = cl2.get_cluster_label(cl.get_cluster_label(state));
    index_of_state_in_sector[state] = sector_members.at(sector_tmp).size();
    sector_of_state[state] = sector_tmp;
    sector_members[sector_tmp].push_back(state);
  }
  dim_sectors.resize(num_sectors_);
  for (int sector = 0; sector < num_sectors_; ++sector) {
    dim_sectors[sector] = sector_members[sector].size();
  }

  // Sanich check
  for (int k = 0; k < ham.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<SCALAR>::InnerIterator it(ham, k); it; ++it) {
      const int dst_state = it.row();
      const int src_state = it.col();
      check_true(sector_of_state[src_state] == sector_of_state[dst_state]);
    }
  }

  //divide Hamiltonin by sector
  ham_sectors.resize(num_sectors_);
  std::vector<int> dummy(num_sectors_);
  split_op_into_sectors(num_sectors_,
                        ham,
                        dim_sectors,
                        index_of_state_in_sector,
                        sector_of_state,
                        dummy.begin(),
                        ham_sectors,
                        true);

  //identify which sectors are connected by a creation/annihilation operator
  sector_connection.resize(boost::extents[2][flavors_][num_sectors_]);
  d_ops_sectors.resize(flavors_);
  ddag_ops_sectors.resize(flavors_);
  std::fill(sector_connection.origin(), sector_connection.origin() + sector_connection.num_elements(), -1);
  for (int flavor = 0; flavor < flavors_; ++flavor) {
    d_ops_sectors[flavor].resize(num_sectors_);
    ddag_ops_sectors[flavor].resize(num_sectors_);

    typedef boost::multi_array_types::index_range range;

    boost::multi_array<int, 3>::array_view<1>::type myview =
        sector_connection[boost::indices[0][flavor][range(0, num_sectors_)]];
    split_op_into_sectors(num_sectors_,
                          ddag_ops[flavor],
                          dim_sectors,
                          index_of_state_in_sector,
                          sector_of_state,
                          myview.origin(),
                          ddag_ops_sectors[flavor]);

    boost::multi_array<int, 3>::array_view<1>::type myview2 =
        sector_connection[boost::indices[1][flavor][range(0, num_sectors_)]];
    split_op_into_sectors(num_sectors_,
                          d_ops[flavor],
                          dim_sectors,
                          index_of_state_in_sector,
                          sector_of_state,
                          myview2.origin(),
                          d_ops_sectors[flavor]);
  }

  sector_connection_reverse.resize(boost::extents[2][flavors_][num_sectors_]);
  std::fill(sector_connection_reverse.origin(),
            sector_connection_reverse.origin() + sector_connection_reverse.num_elements(), -1);
  for (int flavor = 0; flavor < flavors_; ++flavor) {
    for (int op = 0; op < 2; ++op) {
      for (int src_sector = 0; src_sector < num_sectors_; ++src_sector) {
        const int dst_sector = sector_connection[op][flavor][src_sector];
        if (dst_sector != nirvana) {
          sector_connection_reverse[op][flavor][dst_sector] = src_sector;
        }
      }
    }
  }

  // Sanity check
  for (int flavor = 0; flavor < flavors_; ++flavor) {
    for (int k = 0; k < d_ops[flavor].outerSize(); ++k) {
      int count = 0;
      int dst_sector = -1;
      for (typename Eigen::SparseMatrix<SCALAR>::InnerIterator it(d_ops[flavor], k); it; ++it) {
        const int src_sector = sector_of_state[it.col()];
        if (count == 0) {
          dst_sector = sector_connection[0][flavor][src_sector];
        }
        check_true(dst_sector == sector_connection[0][flavor][src_sector]);
      }
    }

    for (int k = 0; k < ddag_ops[flavor].outerSize(); ++k) {
      int count = 0;
      int dst_sector = -1;
      for (typename Eigen::SparseMatrix<SCALAR>::InnerIterator it(ddag_ops[flavor], k); it; ++it) {
        const int src_sector = sector_of_state[it.col()];
        if (count == 0) {
          dst_sector = sector_connection[1][flavor][src_sector];
        }
        check_true(dst_sector == sector_connection[1][flavor][src_sector]);
      }
    }
  }
}


// Compute total particle number for each sector
template<typename SCALAR, typename DERIVED>
void AtomicModel<SCALAR, DERIVED>::init_nelec_sectors() {
  nelec_sectors.resize(num_sectors_);
  for (auto sector = 0; sector < num_sectors_; ++sector) {
    sparse_matrix_t nelec_op(dim_sector(sector), dim_sector(sector));
    for (auto flavor = 0; flavor < flavors_; ++flavor) {
      auto sector_mid = sector_connection[ANNIHILATION_OP][flavor][sector];
      if (sector_mid < 0) {
        continue;
      }
      nelec_op += ddag_ops_sectors[flavor][sector_mid] * d_ops_sectors[flavor][sector];
    }
    double nelec_ = get_real(Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>(nelec_op)(0,0));
    nelec_sectors[sector] = std::round(nelec_);
    check_true(std::abs(nelec_sectors[sector] - nelec_) < 1e-3,
        "Something got wrong in init_nelec_sector!");
  }
}

template<typename SCALAR, typename DERIVED>
template<int N>
void AtomicModel<SCALAR, DERIVED>::apply_op_bra(const EqualTimeOperator<N> &op, BRAKET_T &bra) const {
  for (int i = 0; i < N; ++i) {
    static_cast<const DERIVED *>(this)->apply_op_hyb_bra(CREATION_OP, op.flavor(2 * i), bra);
    static_cast<const DERIVED *>(this)->apply_op_hyb_bra(ANNIHILATION_OP, op.flavor(2 * i + 1), bra);
  }
}

template<typename SCALAR, typename DERIVED>
template<int N>
void AtomicModel<SCALAR, DERIVED>::apply_op_ket(const EqualTimeOperator<N> &op, BRAKET_T &ket) const {
  for (int i = 0; i < N; ++i) {
    static_cast<const DERIVED *>(this)->apply_op_hyb_ket(ANNIHILATION_OP, op.flavor(2 * N - 1 - 2 * i), ket);
    static_cast<const DERIVED *>(this)->apply_op_hyb_ket(CREATION_OP, op.flavor(2 * N - 2 - 2 * i), ket);
  }
}
