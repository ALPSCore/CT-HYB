#pragma once
#include<Eigen/Dense>
#include<Eigen/SparseCore>
#include<vector>
#include<fstream>

#include "../util.hpp"

/*
 * Constructing matrix represantation of creation and annihilation operators in occupation basis
 */
template<typename T>
class FermionOperator {
 public:
  //typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic > matrix_t;
  typedef Eigen::SparseMatrix<T> sparse_matrix_t;
  FermionOperator(int orbitals) :
      orbitals_(orbitals),
      dim_(1 << orbitals_) {
    create_creation_ops();
#ifndef NDEBUG
    validate();
    print_to_file();
#endif
  }
  int orbital_mask(int i) { return 1 << i; }
  int orbitals() const { return orbitals_; }
  int dim() const { return dim_; }
  int cdag(int orbital, int targ, int src) const { return cdag_ops[orbital](src, targ); }
  int c(int orbital, int targ, int src) const { return c_ops[orbital](src, targ); }
  const sparse_matrix_t &get_c(int orbital) const {
    assert(orbital >= 0 && orbital < orbitals_);
    return c_ops[orbital];
  }
  const sparse_matrix_t &get_cdag(int orbital) const {
    assert(orbital >= 0 && orbital < orbitals_);
    return cdag_ops[orbital];
  }

  void print_to_file() {
    for (int i = 0; i < orbitals_; ++i) {
      std::stringstream cname;
      cname << "c_" << i << ".dat";
      std::stringstream cdagname;
      cdagname << "cdag_" << i << ".dat";
      std::ofstream c_file(cname.str().c_str());
      c_file << c_ops[i];
      std::ofstream cdag_file(cdagname.str().c_str());
      cdag_file << cdag_ops[i];
    }
  }
 private:
  void create_creation_ops() {
    for (int i = 0; i < orbitals_; ++i) {
      //annihilator will change source state to target state.
      std::vector<Eigen::Triplet<T> > triplet_list;
      for (int source_state = 0; source_state < dim_; ++source_state) {
        if (source_state & orbital_mask(i)) {
          int target_state = source_state ^orbital_mask(i);
          int permutation_sign = 1;
          for (int k = i + 1; k < orbitals_; ++k) if (source_state & orbital_mask(k)) permutation_sign *= -1;
          triplet_list.push_back(Eigen::Triplet<T>(source_state, target_state, permutation_sign));
          //c_i(source_state, target_state)=permutation_sign;
        }
      }
      sparse_matrix_t c_i(dim_, dim_);
      c_i.setFromTriplets(triplet_list.begin(), triplet_list.end());
      c_ops.push_back(c_i);
      cdag_ops.push_back(c_i.transpose());
    }
  }
  ///check all anticommutation relations
  void validate() const {
    sparse_matrix_t identity(dim_, dim_), zero(dim_, dim_);
    identity.setIdentity();

    for (int i = 0; i < orbitals_; ++i) {
      for (int j = 0; j < orbitals_; ++j) {
        if (maxAbsCoeff(anti_commutator(c_ops[i], c_ops[j])) > 1.e-5)
          throw std::runtime_error("fermionic operators do not behave as expected: c ops");
        if (maxAbsCoeff(anti_commutator(cdag_ops[i], cdag_ops[j])) > 1.e-5)
          throw std::runtime_error("fermionic operators do not behave as expected: cdag ops");
        if (maxAbsCoeff(sparse_matrix_t(anti_commutator(cdag_ops[i], c_ops[j]) - ((i == j) ? identity : zero)))
            > 1.e-5) {
          std::cout << "cdag: " << cdag_ops[i] << std::endl;
          std::cout << "c    : " << c_ops[i] << std::endl;
          std::cout << i << " " << j << std::endl << anti_commutator(cdag_ops[i], c_ops[j]) << std::endl;
          throw std::runtime_error("fermionic operators do not behave as expected.");
        }
      }
    }
  }
  ///the anticommutation relations of fermion operators
  sparse_matrix_t anti_commutator(const sparse_matrix_t &i, const sparse_matrix_t &j) const { return i * j + j * i; }
  ///creation operators: one per (spin-)orbital
  std::vector<sparse_matrix_t> c_ops;
  ///annihilation operators: one per (spin-)orbital
  std::vector<sparse_matrix_t> cdag_ops;
  ///number of orbitals
  const int orbitals_;
  ///dimension of fock space
  const int dim_;
};

