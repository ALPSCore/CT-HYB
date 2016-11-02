#pragma once

#include <vector>
#include <complex>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <utility>

#include <boost/tuple/tuple.hpp>
#include <boost/multi_array.hpp>
#include <boost/format.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <boost/lambda/lambda.hpp>

#include <Eigen/Dense>

#include "hybfermion.hpp"
#include "clustering.hpp"
#include "../util.hpp"
#include "../operator.hpp"
#include "../wide_scalar.hpp"

//forward declaration for alps::params
namespace alps {
namespace params_ns {
class params;
};
using params_ns::params;
}


const int nirvana = -1;

/*
 * A class holding a bra or a ket with some additional information
 * Usually, OBS is a matrix type.
 */
template<typename Scalar, typename OBJ>
class Braket {
 public:
  typedef EXTENDED_REAL norm_type;

  Braket() : sector_(nirvana), obj_(0, 0), coeff_(1.0) { }

  Braket(int sector, const OBJ &obj) {
    sector_ = sector;
    obj_ = obj;
    coeff_ = 1.0;
  }

  //Accessors
  inline int sector() const { return sector_; };
  inline const OBJ &obj() const { return obj_; };
  inline OBJ &obj() { return obj_; };
  inline EXTENDED_REAL coeff() const { return coeff_; };
  inline void set_coeff(const EXTENDED_REAL &coeff) { coeff_ = coeff; };

  inline bool invalid() const {
    return (sector_ == nirvana || size1(obj_) == 0 || size2(obj_) == 0);
  }

  inline void set_invalid() {
    sector_ = nirvana;
    obj_.resize(0, 0);
  }

  inline void set_sector(int sector) {
    sector_ = sector;
  }

  inline void swap_obj(OBJ &obj) {
    using std::swap;
    swap(obj_, obj);
    normalize();
  }

  inline int min_dim() const {
    return std::min(size1(obj_), size2(obj_));
  }

  inline norm_type compute_spectral_norm() {
    normalize();
    norm_type r = invalid() ? norm_type(0.0) : coeff_ * spectral_norm_diag<Scalar>(obj_);
    if (!(r >= 0.0)) {
      std::cout << "comp debug " << coeff_ << " " << spectral_norm_diag<Scalar>(obj_) << std::endl;
      std::cout << "comp prod " << coeff_ * spectral_norm_diag<Scalar>(obj_) << std::endl;
      std::cout << "comp size " << obj_.rows() << " " << obj_.cols() << std::endl;
      std::cout << "comp debug " << obj_ << " === " << std::endl;
      exit(-1);
    }
    assert(r >= 0.0);
    return r;
  }

  inline norm_type max_norm() const {
    if (invalid()) return norm_type(0.0);
    return coeff_ * obj_.cwiseAbs().maxCoeff();
  }

  void normalize() {
    if (invalid()) return;

    double maxval = obj_.cwiseAbs().maxCoeff();
    if (maxval == 0.0) {
      set_invalid();
      return;
    }
    coeff_ *= maxval;
    double rtmp = 1 / maxval;
    for (int j = 0; j < obj_.cols(); ++j) {
      for (int i = 0; i < obj_.rows(); ++i) {
        if (std::abs(obj_(i, j)) < maxval * 1E-30) {
          obj_(i, j) = 0.0;
        } else {
          obj_(i, j) *= rtmp;
        }
      }
    }
  }

 private:
  int sector_;
  OBJ obj_;
  norm_type coeff_;
};

template<class T>
struct model_traits { };

/**
 * @brief Class for the definition of impurity model.
 *
 * This class is abstract and we must use a derived class, e.g., ImpurityModelEigenBasis.
 * This class knows everything about the local Hamiltonian of your impurity model.
 * Several functions define the actual procedures of an imaginary time evolution and application of operators on a bra/ket.
 *
 * In the class ImpurityModel, we construct occupation-basis presentations of the local Hamiltonian and annihilation/creation operators.
 * They are analyzed for splitting the local Hilbert space into symmetry sectors.
 *
 * For the partitioning of the Hilbert space, we use the Hoshen-Kopelman Algorithm (see https://www.ocf.berkeley.edu/~fricke/projects/hoshenkopelman/hoshenkopelman.html).
 * The idea behind is the following.
 * If the matrix of the your local Hamiltonian has only zero elements, each vector in the occupation basis forms its own sector.
 * In other word, the number of symmetry sectors is equal to the dimension of the local Hilbert space.
 * Each time we introduce a new non-zero element into the matrix, it may reduce the symmetry of the model.
 * If the non-zero element connects two vectors belonging to different symmetry sectors, those sector must be merged.
 * The actual procedure of merging sectors is done with the Hoshen-Kopelman Algorithm,
 * which is an efficient algorithm for percolation problems.
 *
 * We achieve polymorphism with the Curiously Recurring Template Pattern (CRTP).
 * This means that the super class ImpurityModel accepts a derived class as a template argument.
 * This allows using virtual functions, which might be expensive.
 *
 * This class holds some information about the hybridization function.
 * TO DO: It might be better to separate it into another class.
 *
 * @param SCALAR scalar type of the Hamiltonian (e.g., double, std::complex<double>)
 * @param DERIVED derived class
 */
template<typename SCALAR, typename DERIVED>
class ImpurityModel {
 public:
  typedef typename ExtendedScalar<SCALAR>::value_type EXTENDED_SCALAR;

  //! Type of a bra and a ket for imaginary time evolution. This may be a matrix type, vector type, etc.
  typedef typename model_traits<DERIVED>::BRAKET_T BRAKET_T;

  //! Container for the hybridization function
  typedef boost::multi_array<SCALAR, 3> hybridization_container_t;

  //! Dense matrix type
  typedef Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

  //! Complex-number dense matrix type (will be removed)
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> complex_matrix_t;

 protected:
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> real_matrix_t;
  typedef Eigen::SparseMatrix<SCALAR> sparse_matrix_t;

 public:
  //! Contruct the impurity model
  ImpurityModel(const alps::params &par, bool verbose = false);
  ImpurityModel(const alps::params &par,
                const std::vector<boost::tuple<int, int, SCALAR> > &nonzero_t_vals_list,
                const std::vector<boost::tuple<int, int, int, int, SCALAR> > &nonzero_U_vals_list,
                bool verbose = false);
  virtual ~ImpurityModel();

  static void define_parameters(alps::params &parameters);

  //! Return the number of flavors (num of sites x num of spins)
  inline int num_flavors() const {
    return flavors_;
  }

  //! Return the number of symmetry sectors found
  inline int num_sectors() const {
    return num_sectors_;
  }

  inline const matrix_t &get_rotmat_Delta() const {
    return rotmat_Delta;
  }

  inline const hybridization_container_t &get_F() const {
    return F;
  }

  /**
   * @brief This function returns the sector that the resultant bra belongs to
   * when an annihilation/creation operator is applied to a bra in the source sector.
   * This function is defined in model.ipp
   *
   * @param op creation/annihilation operator
   * @param flavor flavor of the creation/annihilation operator
   * @param src_sector the sector that the bra belongs to before the operation
   */
  int get_dst_sector_ket(OPERATOR_TYPE op, int flavor, int src_sector) const;

  /**
   * @brief This function returns the sector that the resultant bra belongs to
   * when an annihilation/creation operator is applied to a bra in the source sector from the right-hand side.
   * This function is defined in model.ipp
   *
   * @param op creation/annihilation operator
   * @param flavor flavor of the creation/annihilation operator
   * @param src_sector the sector that the bra belongs to before the operation
   */
  int get_dst_sector_bra(OPERATOR_TYPE op, int flavor, int src_sector) const;

  /**
   * The following functions to be implemented in a derived class
   **/
  double get_reference_energy() const;

  int dim_sector(int sector) const;

  //Apply d and ddag operators for hybridization function on a bra or a ket
  void apply_op_hyb_bra(const OPERATOR_TYPE &op_type, int flavor, BRAKET_T &bra) const;

  void apply_op_hyb_ket(const OPERATOR_TYPE &op_type, int flavor, BRAKET_T &ket) const;

  typename ExtendedScalar<SCALAR>::value_type
      product(const BRAKET_T &bra, const BRAKET_T &ket) const;

  //Apply exp(-t H0) on a bra or a ket
  void sector_propagate_bra(BRAKET_T &bra, double t) const;

  void sector_propagate_ket(BRAKET_T &ket, double t) const;

  int num_brakets() const;

  typename model_traits<DERIVED>::BRAKET_T get_outer_bra(int bra) const;

  typename model_traits<DERIVED>::BRAKET_T get_outer_ket(int ket) const;

  //Min eigenenergy in a given sector
  double min_energy(int sector) const;//deprecated.

  bool translationally_invariant() const;

  /**
   * @brief Apply c^dagger c on a bra from the right hand side.
   */
  template<int N>
  void apply_op_bra(const EqualTimeOperator<N> &op, BRAKET_T &bra) const;

  /**
   * @brief Apply c^dagger c on a ket.
   */
  template<int N>
  void apply_op_ket(const EqualTimeOperator<N> &op, BRAKET_T &ket) const;

 protected:
  const int sites_, spins_, flavors_, dim_, ntau_, Np1_;
  double reference_energy_;
  bool verbose_;

  //for initialization
  void read_U_tensor(const alps::params &par);
  void read_hopping(const alps::params &par);
  void read_hybridization_function(const alps::params &par);
  void read_rotation_hybridization_function(const alps::params &par);
  void hilbert_space_partioning(const alps::params &par);

  //getter
  const sparse_matrix_t &creation_operators_hyb(int flavor, int sector) {
    return ddag_ops_sectors[flavor][sector];
  }
  const sparse_matrix_t &annihilation_operators_hyb(int flavor, int sector) {
    return d_ops_sectors[flavor][sector];
  }
  const std::vector<std::vector<int> > &get_sector_members() const {
    return sector_members;
  }

  //Hamiltonian
  std::vector<sparse_matrix_t> ham_sectors;

  //Index: cdag, cdag, c, c
  std::vector<boost::tuple<int, int, int, int, SCALAR> > nonzero_U_vals;

  //Index: cdag, c
  // t_{ij} cdag_i c_j
  std::vector<boost::tuple<int, int, SCALAR> > nonzero_t_vals;

  boost::multi_array<SCALAR, 4> U_tensor_rot;

  //fermionic operators
  std::vector<std::vector<sparse_matrix_t> > d_ops_sectors, ddag_ops_sectors;//flavor, sector

  //Remember to which sector a vector in a given sector moves
  // when a creation or annihilation operator is applied from the left.
  //index0: creation(=0) or annihilation(=1)
  //index1: flavor
  //index2: sector
  //key: src sector
  //value: target sector
  boost::multi_array<int, 3> sector_connection, sector_connection_reverse;

 private:
//results of partioning of the Hilbert space
  int num_sectors_;

  std::vector<std::vector<int> > sector_members;
  std::vector<int> sector_of_state, index_of_state_in_sector, dim_sectors;

  //Hybridization function
  hybridization_container_t F; // Hybridization Function
  matrix_t rotmat_F, inv_rotmat_F;
  matrix_t rotmat_Delta, inv_rotmat_Delta;
};

/**
 * @brief Definition of local eigenbasis.
 *
 * This defines the eigenbasis of the local impurity model.
 *
 */
template<typename SCALAR>
class ImpurityModelEigenBasis: public ImpurityModel<SCALAR, ImpurityModelEigenBasis<SCALAR> > {
 private:
  typedef ImpurityModel<SCALAR, ImpurityModelEigenBasis<SCALAR> > Base;
  typedef typename Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> dense_matrix_t;
  typedef dense_matrix_t braket_obj_t;

 public:
  typedef Braket<SCALAR, Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> > BRAKET_T;
  using typename Base::EXTENDED_SCALAR;

  ImpurityModelEigenBasis(const alps::params &par, bool verbose = false);
  ImpurityModelEigenBasis
      (const alps::params &par, const std::vector<boost::tuple<int, int, SCALAR> > &nonzero_t_vals_list,
       const std::vector<boost::tuple<int, int, int, int, SCALAR> > &nonzero_U_vals_list, bool verbose = false);
  static void define_parameters(alps::params &parameters);

  void apply_op_hyb_bra(const OPERATOR_TYPE &op_type, int flavor, BRAKET_T &bra) const;
  void apply_op_hyb_ket(const OPERATOR_TYPE &op_type, int flavor, BRAKET_T &ket) const;
  typename ExtendedScalar<SCALAR>::value_type product(const BRAKET_T &bra, const BRAKET_T &ket) const;

  inline int dim_sector(int sector) const {
    assert(sector >= 0 && sector < eigenvals_sector.size());
    return eigenvals_sector[sector].size();
  }
  //Apply exp(-t H0) on a bra or a ket
  void sector_propagate_bra(BRAKET_T &bra, double t) const;
  void sector_propagate_ket(BRAKET_T &ket, double t) const;
  typename model_traits<ImpurityModelEigenBasis<SCALAR> >::BRAKET_T get_outer_bra(int bra) const;
  typename model_traits<ImpurityModelEigenBasis<SCALAR> >::BRAKET_T get_outer_ket(int ket) const;

  inline double min_energy(int sector) const {
    assert(sector >= 0 && sector < Base::num_sectors());
    return min_eigenval_sector[sector];
  }

  inline int num_brakets() const {
    return num_braket_;
  }

  bool translationally_invariant() const;

 private:
  void build_basis(const alps::params &par);
  void build_outer_braket(const alps::params &par);
  //for debug
  void check_evecs(const std::vector<dense_matrix_t> ham_sector, const std::vector<dense_matrix_t> &evecs_sector);
  bool is_sector_active(int sector) const;
  std::vector<std::vector<double> > eigenvals_sector;
  std::vector<double> min_eigenval_sector;
  std::vector<std::vector<dense_matrix_t> > ddag_ops_eigen, d_ops_eigen;//flavor, sector

  int num_braket_;
  //equal to the number of active sectors
  std::vector<BRAKET_T> bra_list, ket_list;
};

template<typename SCALAR>
struct model_traits<ImpurityModelEigenBasis<SCALAR> > {
  typedef SCALAR SCALAR_T;
  typedef Braket<SCALAR, Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> > BRAKET_T;
};

//inline double compute_exp(double a) {
//const double limit = std::log(std::numeric_limits<double>::min())/2;
//if (a < limit) {
//return 0.0;
//} else {
//return std::exp(a);
//}
//}

/**
 * Compute exp(-t*energy) in an elementray-wise fashion.
 * Small values much smaller than the largest element will be set to zero for numerical stability.
 * (The cut off is exp(-60.0) ~ 10^{-30})
 */
inline double compute_exp_vector_safe(const double tau,
                                      const std::vector<double> &energies,
                                      std::vector<double> &exp_a) {
  std::vector<double> a(energies.size());
  for (int i = 0; i < energies.size(); ++i) {
    a[i] = -tau * energies[i];
  }
  const double max_val = *std::max_element(a.begin(), a.end());
  exp_a.resize(a.size());
  for (int i = 0; i < a.size(); ++i) {
    double da = a[i] - max_val;
    if (da < -60.0) {
      exp_a[i] = 0.0;
    } else {
      exp_a[i] = std::exp(da);
    }
  }
  return std::exp(max_val);
}

typedef ImpurityModelEigenBasis<double> REAL_EIGEN_BASIS_MODEL;
typedef ImpurityModelEigenBasis<std::complex<double> > COMPLEX_EIGEN_BASIS_MODEL;
