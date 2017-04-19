#pragma once

#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <assert.h>

#include <boost/math/special_functions/bessel.hpp>
#include <boost/multi_array.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>

#include <alps/gf_extension/ir_basis.hpp>

/**
 * Class represents an orthogonal basis set.
 */
class OrthogonalBasis {
 public:
  OrthogonalBasis() {};

  /** Returns <l|l> where |l> is the l-th basis vector. */
  virtual double norm2(int l) const = 0;

  /** Returns the dimension of the basis set */
  virtual int dim() const = 0;

  /** Returns values at x (-1 < x < 1) */
  virtual void value(double x, std::vector<double> &val) const = 0;

  /** Returns the transformation matrix to positive Matsubara frequencies */
  virtual const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Tnl(int n_iw) const = 0;

};

/**
 * Kernel IR basis
 */
template<class Base>
class IrBasis : public OrthogonalBasis {
 public:
  IrBasis(double Lambda, int max_dim) {
    p_basis_.reset(new Base(Lambda, max_dim, 1e-8, 501));
  }

  double norm2(int l) const {return 1.0;}
  int dim() const {return p_basis_->dim();}
  void value(double x, std::vector<double> &val) const {
    p_basis_->value(x, val);
  }

  /** Returns the transformation matrix to positive Matsubara frequencies */
  virtual const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Tnl(int n_iw) const;

  alps::gf::numerical_mesh<double>
  construct_mesh(double beta) const {
    return alps::gf::numerical_mesh<double>{beta, p_basis_->all(), p_basis_->get_statistics()};
  }

 private:
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> complex_matrix_t;
  boost::shared_ptr<Base> p_basis_;
  mutable complex_matrix_t Tnl_;
};


typedef IrBasis<alps::gf_extension::fermionic_ir_basis> FermionicIRBasis;
typedef IrBasis<alps::gf_extension::bosonic_ir_basis> BosonicIRBasis;
