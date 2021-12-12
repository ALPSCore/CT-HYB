#pragma once

#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <assert.h>

#include <boost/multi_array.hpp>
#include "irbasis.hpp"

enum STATISTICS {
  FERMION,
  BOSON
};


/**
 * Class represents an orthogonal basis set.
 * 
 *  G(\tau) = \sum_{l=0} G_l U_l(\tau)
 */
class OrthogonalBasis {
 public:
  OrthogonalBasis() {};

  /** Returns \int d\tau |U_l(\tau)|^2 **/
  virtual double norm2(int l) const = 0;

  /** Returns the dimension of the basis set */
  virtual int dim() const = 0;

  /** Returns values U_l(\tau) at tau (0 <= tau <= beta) */
  virtual void value(double tau, std::vector<double> &val) const = 0;
};

/**
 * Legendre basis
 * U_l(tau) = (\sqrt(2*l+1)/beta) * P_l[x(\tau)]
 */
class LegendreBasis : public OrthogonalBasis {
 public:
  LegendreBasis(STATISTICS stat, double beta, int size) :
    stat_(stat),
    beta_(beta),
    size_(size),
    norm2_(1/beta_),
    coeff_l_(size)
  {
    for (auto l=0; l<size; ++l) {
      coeff_l_[l] = std::sqrt(2*l+1)/beta_;
    }
  }

  inline double norm2(int l) const {return norm2_;}
  inline int dim() const {return size_;}

  inline void value(double tau, std::vector<double> &val) const {
    auto x = 2*tau/beta_ - 1;

    // Compute P_l[x(tau)]
    for (auto l = 0; l < size_; l++) {
      if (l == 0) {
        val[l] = 1;
      } else if (l == 1) {
        val[l] = x;
      } else {
        val[l] = ((2 * l - 1) * x * val[l - 1] - (l - 1) * val[l - 2])/l;//l
      }
    }

    // Multiply by sqrt(2*l+1)/beta
    for (auto l=0; l<size_; ++l) {
      val[l] *= coeff_l_[l];
    }
  }

 private:
  STATISTICS stat_;
  double beta_;
  int size_;
  double coeff_;
  double norm2_;
  std::vector<double> coeff_l_;
};

/**
 * IR basis
 */
class IrBasis : public OrthogonalBasis {
 public:
  IrBasis(STATISTICS stat, double beta, double Lambda, int max_dim, std::string data_file="") :
    stat_(stat),
    beta_(beta),
    size_(0),
    coeff_(std::sqrt(2/beta))
  {
    std::vector<double> valid_Lambda {1e+1, 1e+2, 1e+3, 1e+4, 1e+5, 1e+6, 1e+7};
    if (std::find(valid_Lambda.begin(), valid_Lambda.end(), Lambda) == valid_Lambda.end()) {
      throw std::runtime_error("Invalid Lambda!");
    }
    if (data_file=="") {
      data_file = "__INSTALL_PREFIX__/share/irbasis.h5";
    }
    auto basis = irbasis::load("F", Lambda, data_file);
    p_basis_ = std::shared_ptr<irbasis::basis>(&basis);
    if (max_dim > p_basis_->dim()) {
      max_dim = p_basis_->dim();
    }
    size_ = max_dim;
  }

  inline double norm2(int l) const {return 1.0;}
  inline int dim() const {return size_;}

  inline void value(double tau, std::vector<double> &val) const {
    auto x = 2*tau/beta_ - 1;
    if (val.size() != size_) {
      throw std::runtime_error("Invalid size of val!");
    }
    for (auto l=0; l<size_; ++l) {
      val[l] = coeff_ * p_basis_->ulx(l, x);
    }
  }

 private:
  STATISTICS stat_;
  std::shared_ptr<irbasis::basis> p_basis_;
  double beta_;
  int size_;
  double coeff_;
};