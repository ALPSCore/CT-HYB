#include "model.hpp"

template<typename T>
struct PruneHelper {
  PruneHelper(double eps) : eps_(eps) { };
  bool operator()(const int &row, const int &col, const T &value) const {
    return (std::abs(value) > eps_);
  }
  const double eps_;
};

template<typename SCALAR, typename DERIVED>
ImpurityModel<SCALAR, DERIVED>::ImpurityModel(const alps::params &par, bool verbose)
    : sites_(par["model.sites"]),
      spins_(par["model.spins"]),
      flavors_(sites_ * spins_),
      dim_(1 << flavors_),
      ntau_(static_cast<int>(par["model.n_tau_hyb"])),
      Np1_(ntau_ + 1),
      reference_energy_(-1E+100),//this should be set in a derived class,
      verbose_(verbose),
      U_tensor_rot(boost::extents[flavors_][flavors_][flavors_][flavors_]) {
  read_U_tensor(par);
  read_hopping(par);
  read_hybridization_function(par);
  read_rotation_hybridization_function(par);
  hilbert_space_partioning(par);
}

template<typename SCALAR, typename DERIVED>
void ImpurityModel<SCALAR, DERIVED>::define_parameters(alps::params &parameters) {
  parameters
      .define<std::string>("model.coulomb_tensor_input_file", "Input file containing nonzero elements of U tensor")
      .define<std::string>("model.hopping_matrix_input_file", "Input file for hopping matrix")
      .define<std::string>("model.delta_input_file", "", "Input file for hybridization function Delta(tau)")
      .define<std::string>("model.basis_input_file", "", "Input file for single-particle basis for expansion")
      .define<double>("model.inner_outer_cutoff_energy", 0.1 * std::numeric_limits<double>::max(),
                      "Cutoff energy for inner states for computing trace (measured from the lowest eigenvalue)")
      .define<double>("model.outer_cutoff_energy", 0.1 * std::numeric_limits<double>::max(),
                      "Cutoff energy for outer states for computing trace (measured from the lowest eigenvalue)")
      .define<double>("model.cutoff_ham", 1E-12,
                      "Cutoff for entries in the local Hamiltonian matrix")
      .define<bool>("model.command_line_mode", false,
                    "if you pass Coulomb tensor, hopping matrix, delta tau via parameters instead of using text files [ADVANCED]")
      .define<std::vector<double> >("model.coulomb_tensor_Re", "Real part of U tensor [ADVANCED]")
      .define<std::vector<double> >("model.coulomb_tensor_Im", "Imaginary part of U tensor [ADVANCED]")
      .define<std::vector<double> >("model.hopping_matrix_Re", "Real part of hopping matrix [ADVANCED]")
      .define<std::vector<double> >("model.hopping_matrix_Im", "Imaginary part of hopping matrix [ADVANCED]")
      .define<std::vector<double> >("model.delta_Re", "Real part of delta(tau) [ADVANCED]")
      .define<std::vector<double> >("model.delta_Im", "Imaginary part of delta(tau) [ADVANCED]");
}


//mainly for unitest
template<typename SCALAR, typename DERIVED>
ImpurityModel<SCALAR, DERIVED>::ImpurityModel(const alps::params &par,
                                              const std::vector<boost::tuple<int, int, SCALAR> > &nonzero_t_vals_list,
                                              const std::vector<boost::tuple<int,
                                                                             int,
                                                                             int,
                                                                             int,
                                                                             SCALAR> > &nonzero_U_vals_list,
                                              bool verbose)
    : sites_(par["model.sites"]),
      spins_(par["model.spins"]),
      flavors_(sites_ * spins_),
      dim_(1 << flavors_),
      ntau_(static_cast<int>(par["model.n_tau_hyb"])),
      Np1_(ntau_ + 1),
      verbose_(verbose),
      nonzero_U_vals(nonzero_U_vals_list),
      nonzero_t_vals(nonzero_t_vals_list),
      U_tensor_rot(boost::extents[flavors_][flavors_][flavors_][flavors_]) {
  read_hybridization_function(par);
  read_rotation_hybridization_function(par);
  hilbert_space_partioning(par);
}


template<typename SCALAR, typename DERIVED>
ImpurityModel<SCALAR, DERIVED>::~ImpurityModel() { }

template<typename SCALAR, typename DERIVED>
int ImpurityModel<SCALAR, DERIVED>::get_dst_sector_ket(OPERATOR_TYPE op, int flavor, int src_sector) const {
  assert(flavor >= 0 && flavor < num_flavors());
  assert(src_sector >= 0 && src_sector < num_sectors());
  return sector_connection[static_cast<int>(op)][flavor][src_sector];
}

template<typename SCALAR, typename DERIVED>
int ImpurityModel<SCALAR, DERIVED>::get_dst_sector_bra(OPERATOR_TYPE op, int flavor, int src_sector) const {
  assert(flavor >= 0);
  assert(flavor < num_flavors());
  assert(src_sector >= 0);
  assert(src_sector < num_sectors());
  return sector_connection_reverse[static_cast<int>(op)][flavor][src_sector];
}

template<typename SCALAR, typename DERIVED>
void ImpurityModel<SCALAR, DERIVED>::read_U_tensor(const alps::params &par) {
  if (par["model.command_line_mode"]) {
    const std::vector<double> &Uijkl_Re = par["model.coulomb_tensor_Re"].template as<std::vector<double> >();
    const std::vector<double> &Uijkl_Im = par["model.coulomb_tensor_Im"].template as<std::vector<double> >();
    const int nf4 = num_flavors()*num_flavors()*num_flavors()*num_flavors();
    if (Uijkl_Re.size() != nf4 || Uijkl_Im.size() != nf4) {
      throw std::runtime_error("The size of the Coulomb tensor is wrong");
    }
    int idx = 0;
    for (int f1 = 0; f1 < num_flavors(); ++f1) {
      for (int f2 = 0; f2 < num_flavors(); ++f2) {
        for (int f3 = 0; f3 < num_flavors(); ++f3) {
          for (int f4 = 0; f4 < num_flavors(); ++f4) {
            std::complex<double> ztmp = std::complex<double>(Uijkl_Re[idx], Uijkl_Im[idx]);
            if (std::abs(ztmp) != 0.0) {
              nonzero_U_vals.push_back(boost::make_tuple(f1, f2, f3, f4, mycast<SCALAR>(0.5*ztmp)));
            }
            ++ idx;
          }
        }
      }
    }
  } else if (par.defined("model.coulomb_tensor_input_file")) {
    std::ifstream infile_f(boost::lexical_cast<std::string>(par["model.coulomb_tensor_input_file"]).c_str());
    if (!infile_f.is_open()) {
      std::cerr << "We cannot open " << par["model.coulomb_tensor_input_file"] << "!" << std::endl;
      exit(1);
    }
    if (verbose_) {
      std::cout << "Reading " << par["model.coulomb_tensor_input_file"] << "..." << std::endl;
    }

    int num_elem;
    infile_f >> num_elem;
    if (num_elem < 0) {
      std::runtime_error("The number of elements in U_TENSOR_INPUT_FILE cannot be negative!");
    }
    if (verbose_) {
      std::cout << "Number of non-zero elements in U tensor is " << num_elem << std::endl;
    }

    nonzero_U_vals.reserve(num_elem);
    for (int i_elem = 0; i_elem < num_elem; ++i_elem) {
      double re, im;
      int line, f0, f1, f2, f3;
      infile_f >> line >> f0 >> f1 >> f2 >> f3 >> re >> im;
      if (line != i_elem) {
        throw std::runtime_error(boost::str(boost::format("First column of line %1% is incorrect.") % i_elem));
      }
      if (f0 < 0 || f0 >= flavors_) {
        throw std::runtime_error(boost::str(boost::format("Second column of line %1% is incorrect.") % i_elem));
      }
      if (f1 < 0 || f1 >= flavors_) {
        throw std::runtime_error(boost::str(boost::format("Third column of line %1% is incorrect.") % i_elem));
      }
      if (f2 < 0 || f2 >= flavors_) {
        throw std::runtime_error(boost::str(boost::format("Fourth column of line %1% is incorrect.") % i_elem));
      }
      if (f3 < 0 || f3 >= flavors_) {
        throw std::runtime_error(boost::str(boost::format("Fifth column of line %1% is incorrect.") % i_elem));
      }
      const SCALAR uval = 0.5 * mycast<SCALAR>(std::complex<double>(re, im));
      nonzero_U_vals.push_back(boost::make_tuple(f0, f1, f2, f3, uval));
    }

    infile_f.close();
  } else if (par.defined("model.onsite_U")) {
    if (spins_ == 2) {
      const double uval = par["model.onsite_U"];
      for (int site = 0; site < sites_; ++site) {
        nonzero_U_vals.push_back(boost::make_tuple(2 * site, 2 * site + 1, 2 * site + 1, 2 * site, 0.5 * uval));
      }
    }
  }
}

template<typename SCALAR, typename DERIVED>
void ImpurityModel<SCALAR, DERIVED>::read_hopping(const alps::params &par) {
  if (par["model.command_line_mode"]) {
    const int len = num_flavors() * num_flavors();
    const std::vector<double> &hopping_matrix_Re = par["model.hopping_matrix_Re"].template as<std::vector<double> >();
    const std::vector<double> &hopping_matrix_Im = par["model.hopping_matrix_Im"].template as<std::vector<double> >();
    if (hopping_matrix_Re.size() != len || hopping_matrix_Im.size() != len) {
      throw std::runtime_error("Size of hopping matrix is wrong!");
    }
    int idx = 0;
    for (int f0 = 0; f0 < flavors_; ++f0) {
      for (int f1 = 0; f1 < flavors_; ++f1) {
        const SCALAR hopping = mycast<SCALAR>(
            std::complex<double>(hopping_matrix_Re[idx], hopping_matrix_Im[idx])
        );
        if (std::abs(hopping) != 0.0) {
          nonzero_t_vals.push_back(boost::make_tuple(f0, f1, hopping));
        }
        ++ idx;
      }
    }
  } else if (par.defined("model.hopping_matrix_input_file")) {
    std::ifstream infile_f(boost::lexical_cast<std::string>(par["model.hopping_matrix_input_file"]).c_str());
    if (!infile_f.is_open()) {
      std::cerr << "We cannot open " << par["model.hopping_matrix_input_file"] << "!" << std::endl;
      exit(1);
    }

    //int num_elem;
    //infile_f >> num_elem;
    //if (num_elem<0) {
    //std::runtime_error("The number of elements in HOPPING_MATRIX_INPUT_FILE cannot be negative!");
    //}

    nonzero_t_vals.resize(0);
    //for (int i_elem=0; i_elem<num_elem; ++i_elem) {
    int line = 0;
    for (int f0 = 0; f0 < flavors_; ++f0) {
      for (int f1 = 0; f1 < flavors_; ++f1) {
        double re, im;
        int f0_in, f1_in;
        infile_f >> f0_in >> f1_in >> re >> im;
        if (f0 != f0_in) {
          throw std::runtime_error(boost::str(boost::format("First column of line %1% is incorrect.") % line));
        }
        if (f1 != f1_in) {
          throw std::runtime_error(boost::str(boost::format("Second column of line %1% is incorrect.") % line));
        }
        const SCALAR hopping = mycast<SCALAR>(std::complex<double>(re, im));
        if (std::abs(hopping) != 0.0) {
          nonzero_t_vals.push_back(boost::make_tuple(f0, f1, hopping));
        }
        ++line;
      }
    }
    infile_f.close();
  }
}


template<typename SCALAR, typename DERIVED>
void ImpurityModel<SCALAR, DERIVED>::read_hybridization_function(const alps::params &par) {
  F.resize(boost::extents[flavors_][flavors_][Np1_]);

  if (par["model.command_line_mode"]) {
    std::fill(F.origin(), F.origin() + F.num_elements(), 0.0);
    const std::vector<double> &delta_Re = par["model.delta_Re"].template as<std::vector<double> >();
    const std::vector<double> &delta_Im = par["model.delta_Im"].template as<std::vector<double> >();
    if (delta_Re.size() != F.num_elements() || delta_Im.size() != F.num_elements()) {
      throw std::runtime_error("Size of delta tau is wrong");
    }
    int idx = 0;
    for (int time = 0; time < Np1_; ++time) {
      for (int f1 = 0; f1 < flavors_; ++f1) {
        for (int f2 = 0; f2 < flavors_; ++f2) {
          assert(f2 < F.shape()[0]);
          assert(f1 < F.shape()[1]);
          assert(Np1_ - time - 1 < F.shape()[2]);
          assert(idx < delta_Re.size());
          assert(idx < delta_Im.size());
          F[f2][f1][Np1_ - time - 1] = - mycast<SCALAR>(std::complex<double>(delta_Re[idx], delta_Im[idx]));
          ++ idx;
        }
      }
    }
  } else if (par["model.delta_input_file"].template as<std::string>() == "") {
    std::fill(F.origin(), F.origin() + F.num_elements(), 0.0);
    for (int i = 0; i < flavors_; i++) {
      for (int time = 0; time < Np1_; time++) {
        const double rtmp = time / static_cast<double>(ntau_);
        F[i][i][time] = 0.5 - 2.0 * std::pow(rtmp - 0.5, 2.0);
      }
    }
  } else {
    // read hybridization function from input file with FLAVORS+1 colums \tau, G_1_up, G_1_down, G_2_up ..., G_SITES_down)
    std::ifstream infile_f(par["model.delta_input_file"].template as<std::string>().c_str());
    if (!infile_f.is_open()) {
      std::cerr << "Input file for F cannot be opened!" << std::endl;
      exit(1);
    }

    double real, imag;
    int dummy_it, dummy_i, dummy_j;

    for (int time = 0; time < Np1_; time++) {
      for (int i = 0; i < flavors_; i++) {
        for (int j = 0; j < flavors_; j++) {
          infile_f >> dummy_it >> dummy_i >> dummy_j >> real >> imag;
          if (dummy_it != time) {
            throw std::runtime_error("Format of " + boost::lexical_cast<std::string>(par["model.delta_input_file"]) +
                " is wrong. The value at the first colum should be " +
                boost::lexical_cast<std::string>(time) + "Error at line " +
                boost::lexical_cast<std::string>(time + 1) + ".");
          }
          if (dummy_i != i) {
            throw std::runtime_error("Format of " + boost::lexical_cast<std::string>(par["model.delta_input_file"]) +
                " is wrong. The value at the second colum should be " +
                boost::lexical_cast<std::string>(i) + "Error at line " +
                boost::lexical_cast<std::string>(time + 1) + ".");
          }
          if (dummy_j != j) {
            throw std::runtime_error("Format of " + boost::lexical_cast<std::string>(par["model.delta_input_file"]) +
                " is wrong. The value at the third colum should be " +
                boost::lexical_cast<std::string>(j) + "Error at line " +
                boost::lexical_cast<std::string>(time + 1) + ".");
          }
          //F_ij(tau) = - Delta_ji (beta - tau)
          F[j][i][Np1_ - time - 1] = - mycast<SCALAR>(std::complex<double>(real, imag));
        }
      }
    }
  }
}

template<typename SCALAR, typename DERIVED>
void ImpurityModel<SCALAR, DERIVED>::read_rotation_hybridization_function(const alps::params &par) {
  rotmat_F.resize(flavors_, flavors_);
  inv_rotmat_F.resize(flavors_, flavors_);
  rotmat_Delta.resize(flavors_, flavors_);
  inv_rotmat_Delta.resize(flavors_, flavors_);
  if (!par.defined("model.basis_input_file") || par["model.basis_input_file"] == std::string("")) {
    rotmat_F.setIdentity();
    inv_rotmat_F.setIdentity();
    rotmat_Delta.setIdentity();
    inv_rotmat_Delta.setIdentity();
  } else {
    if (verbose_) {
      std::cout << "Opening " << par["model.basis_input_file"].template as<std::string>() << "..." << std::endl;
    }
    std::ifstream infile_f(par["model.basis_input_file"].template as<std::string>().c_str());
    if (!infile_f.is_open()) {
      std::cerr << "in file for BASIS_INPUT_FILE not open! " << std::endl;
      exit(1);
    }

#ifndef NDEBUG
    std::cout << "Reading " << boost::lexical_cast<std::string>(par["model.basis_input_file"]) << "..." << std::endl;
#endif
    for (int i = 0; i < flavors_; ++i) {
      for (int j = 0; j < flavors_; j++) {
        int i_dummy, j_dummy;
        double real, imag;
        infile_f >> i_dummy >> j_dummy >> real >> imag;
        if (i_dummy != i || j_dummy != j) {
          throw std::runtime_error("Wrong format: BASIS_INPUT_FILE");
        }
        rotmat_F(i, j) = mycast<SCALAR>(std::complex<double>(real, -imag));
        //Caution: the minus in the imaginary part
        //This is because the input is a rotation matrix for Delta(tau) not F(tau).
      }
    }
#ifndef NDEBUG
    std::cout << "Rotation matrix (read)" << std::endl;
    for (int i = 0; i < flavors_; ++i) {
      for (int j = 0; j < flavors_; ++j) {
        std::cout << i << " " << j << " " << rotmat_F(i, j) << std::endl;
      }
    }
#endif

    //normalization
    for (int j = 0; j < flavors_; ++j) {
      double rtmp = 0.0;
      for (int i = 0; i < flavors_; ++i) {
        rtmp += std::abs(rotmat_F(i, j)) * std::abs(rotmat_F(i, j));
      }
      rtmp = 1.0 / std::sqrt(rtmp);
      for (int i = 0; i < flavors_; ++i) {
        rotmat_F(i, j) *= rtmp;
      }
    }

#ifndef NDEBUG
    std::cout << "Rotation matrix (normalized)" << std::endl;
    for (int i = 0; i < flavors_; ++i) {
      for (int j = 0; j < flavors_; ++j) {
        std::cout << i << " " << j << " " << rotmat_F(i, j) << std::endl;
      }
    }
#endif

    //check if the matrix is unitary.
    for (int i = 0; i < flavors_; ++i) {
      for (int j = 0; j < i; ++j) {
        SCALAR rtmp = 0.0;
        for (int k = 0; k < flavors_; ++k) {
          rtmp += myconj(rotmat_F(k, i)) * rotmat_F(k, j);
        }
        if (std::abs(rtmp) > 1e-8) {
          std::cerr << "Orthogonality error in columns " << i << " " << j << std::endl;
          exit(1);
        }
      }
    }

    inv_rotmat_F = rotmat_F.inverse();

    //Rotation matrix for Delta (complex conjugate of rotmat_F).
    //This will be used to transform G back to the original base.
    rotmat_Delta = rotmat_F.conjugate();
    inv_rotmat_Delta = rotmat_Delta.inverse();

#ifndef NDEBUG
    {
      matrix_t should_be_identity = inv_rotmat_F * rotmat_F;
      bool OK = true;
      const double eps = 1e-8;
      for (int i = 0; i < flavors_; ++i) {
        for (int j = 0; j < flavors_; ++j) {
          if (i == j) {
            if (std::abs(should_be_identity(i, j) - 1.0) > eps) {
              OK = false;
            }
          } else {
            if (std::abs(should_be_identity(i, j)) > eps) {
              OK = false;
            }
          }
        }
      }
      assert(OK);
    }
#endif

    matrix_t mattmp(flavors_, flavors_), mattmp2(flavors_, flavors_);
    for (int time = 0; time < Np1_; ++time) {
      for (int iflavor = 0; iflavor < flavors_; ++iflavor) {
        for (int jflavor = 0; jflavor < flavors_; ++jflavor) {
          mattmp(iflavor, jflavor) = F[iflavor][jflavor][time];
        }
      }
      //mattmp.matrix_right_multiply(rotmat_F, mattmp2);
      //inv_rotmat_F.matrix_right_multiply(mattmp2, mattmp);
      mattmp2 = mattmp * rotmat_F;
      mattmp = inv_rotmat_F * mattmp2;
      for (int iflavor = 0; iflavor < flavors_; ++iflavor) {
        for (int jflavor = 0; jflavor < flavors_; ++jflavor) {
          F[iflavor][jflavor][time] = mattmp(iflavor, jflavor);
        }
      }
    }
  }
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
void ImpurityModel<SCALAR, DERIVED>::hilbert_space_partioning(const alps::params &par) {
  const double eps_numerics = 1E-12;
  const double eps = par["model.cutoff_ham"];

  //Compute U tensor in the rotated basis
  //very naive implementation. One might vectorize the code and use the cache...
  matrix_t rotmat_Delta_trans = rotmat_Delta.transpose();
  std::fill(U_tensor_rot.origin(), U_tensor_rot.origin() + U_tensor_rot.num_elements(), 0.0);
  for (int elem = 0; elem < nonzero_U_vals.size(); ++elem) {
    const int a = boost::get<0>(nonzero_U_vals[elem]);
    const int b = boost::get<1>(nonzero_U_vals[elem]);
    const int ap = boost::get<2>(nonzero_U_vals[elem]);
    const int bp = boost::get<3>(nonzero_U_vals[elem]);
    const SCALAR uval = boost::get<4>(nonzero_U_vals[elem]);

    for (int flavor0 = 0; flavor0 < flavors_; ++flavor0) {
      for (int flavor1 = 0; flavor1 < flavors_; ++flavor1) {
        for (int flavor2 = 0; flavor2 < flavors_; ++flavor2) {
          for (int flavor3 = 0; flavor3 < flavors_; ++flavor3) {
            U_tensor_rot[flavor0][flavor1][flavor2][flavor3] +=
                uval * myconj(rotmat_Delta_trans(flavor0, a)) *
                    myconj(rotmat_Delta_trans(flavor1, b)) *
                    rotmat_Delta_trans(flavor2, ap) *
                    rotmat_Delta_trans(flavor3, bp);
          }
        }
      }
    }
  }

  //Compute hopping matrix in the rotated basis
  matrix_t hopping_org_basis(flavors_, flavors_);
  hopping_org_basis.setZero();
  for (int elem = 0; elem < nonzero_t_vals.size(); ++elem) {
    const int i = boost::get<0>(nonzero_t_vals[elem]);
    const int j = boost::get<1>(nonzero_t_vals[elem]);
    const SCALAR hopping = boost::get<2>(nonzero_t_vals[elem]);
    hopping_org_basis(i, j) = hopping;
  }
  for (int flavor = 0; flavor < flavors_; ++flavor) {
    for (int flavor2 = 0; flavor2 < flavors_; ++flavor2) {
      if (hopping_org_basis(flavor, flavor2) != myconj<SCALAR>(hopping_org_basis(flavor2, flavor))) {
        throw std::runtime_error("Error: Hopping matrix is not hermite!");
      }
    }
  }
  matrix_t hopping_rot = rotmat_Delta.adjoint() * hopping_org_basis * rotmat_Delta;

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
  for (int flavor1 = 0; flavor1 < flavors_; ++flavor1) {
    for (int flavor2 = 0; flavor2 < flavors_; ++flavor2) {
      for (int flavor3 = 0; flavor3 < flavors_; ++flavor3) {
        for (int flavor4 = 0; flavor4 < flavors_; ++flavor4) {
          const SCALAR uval = U_tensor_rot[flavor1][flavor2][flavor3][flavor4];
          if (std::abs(uval) > eps_numerics) {
            ham += uval * ddag_ops[flavor1] * ddag_ops[flavor2] * d_ops[flavor3] * d_ops[flavor4];
          }
        }
      }
    }
  }
  for (int flavor2 = 0; flavor2 < flavors_; ++flavor2) {
    for (int flavor1 = 0; flavor1 < flavors_; ++flavor1) {
      if (std::abs(hopping_rot(flavor1, flavor2)) > eps_numerics) {
        ham += hopping_rot(flavor1, flavor2) * ddag_ops[flavor1] * d_ops[flavor2];
      }
    }
  }
  ham.prune(PruneHelper<SCALAR>(eps));

  //Partionining of Hilbert space according to symmetry
  Clustering cl(dim_);
  for (int k = 0; k < ham.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<SCALAR>::InnerIterator it(ham, k); it; ++it) {
      cl.connect_vertices(it.row(), it.col());
    }
  }
  cl.finalize_labeling();

  if (verbose_) {
    std::cout << "dim of Hilbert space " << dim_ << std::endl;
    std::cout << "# of blocks " << cl.get_num_clusters() << std::endl;
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
    std::cout << "# of sectors " << cl2.get_num_clusters() << std::endl;
  }

  //identity members in each symmetry sector
  num_sectors_ = cl2.get_num_clusters();
  sector_members.resize(num_sectors_);
  sector_of_state.resize(dim_);
  index_of_state_in_sector.resize(dim_);
  for (int state = 0; state < dim_; ++state) {
    int sector_tmp = cl2.get_cluster_label(cl.get_cluster_label(state));
    assert(sector_tmp < num_sectors_);
    index_of_state_in_sector[state] = sector_members[sector_tmp].size();
    sector_of_state[state] = sector_tmp;
    sector_members[sector_tmp].push_back(state);
  }
  dim_sectors.resize(num_sectors_);
  for (int sector = 0; sector < num_sectors_; ++sector) {
    dim_sectors[sector] = sector_members[sector].size();
  }

#ifndef NDEBUG
  for (int k = 0; k < ham.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<SCALAR>::InnerIterator it(ham, k); it; ++it) {
      const int dst_state = it.row();
      const int src_state = it.col();
      assert(sector_of_state[src_state] == sector_of_state[dst_state]);
    }
  }
#endif

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

#ifndef NDEBUG
  for (int flavor = 0; flavor < flavors_; ++flavor) {
    for (int k = 0; k < d_ops[flavor].outerSize(); ++k) {
      int count = 0;
      int dst_sector = -1;
      for (typename Eigen::SparseMatrix<SCALAR>::InnerIterator it(d_ops[flavor], k); it; ++it) {
        const int src_sector = sector_of_state[it.col()];
        if (count == 0) {
          dst_sector = sector_connection[0][flavor][src_sector];
        }
        assert(dst_sector == sector_connection[0][flavor][src_sector]);
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
        assert(dst_sector == sector_connection[1][flavor][src_sector]);
      }
    }
  }
#endif
}

template<typename SCALAR, typename DERIVED>
template<int N>
void ImpurityModel<SCALAR, DERIVED>::apply_op_bra(const EqualTimeOperator<N> &op, BRAKET_T &bra) const {
  for (int i = 0; i < N; ++i) {
    static_cast<const DERIVED *>(this)->apply_op_hyb_bra(CREATION_OP, op.flavor(2 * i), bra);
    static_cast<const DERIVED *>(this)->apply_op_hyb_bra(ANNIHILATION_OP, op.flavor(2 * i + 1), bra);
  }
}

template<typename SCALAR, typename DERIVED>
template<int N>
void ImpurityModel<SCALAR, DERIVED>::apply_op_ket(const EqualTimeOperator<N> &op, BRAKET_T &ket) const {
  for (int i = 0; i < N; ++i) {
    static_cast<const DERIVED *>(this)->apply_op_hyb_ket(ANNIHILATION_OP, op.flavor(2 * N - 1 - 2 * i), ket);
    static_cast<const DERIVED *>(this)->apply_op_hyb_ket(CREATION_OP, op.flavor(2 * N - 2 - 2 * i), ket);
  }
}
