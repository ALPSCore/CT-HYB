#pragma once

#include<complex>
#include<cmath>
#include<vector>
#include<assert.h>

#include "boost/math/special_functions/bessel.hpp"
#include "boost/multi_array.hpp"

#include<Eigen/Dense>

class LegendreTransformer {
 public:
  LegendreTransformer(int n_matsubara, int n_legendre);

 private:
  const int n_matsubara_, n_legendre_;

 public:
  const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &Tnl() const;
  void compute_legendre(double x, std::vector<double> &val) const;
  void compute_legendre(const std::vector<double> &xval, boost::multi_array<double, 2> &val) const;
  const std::vector<double> &get_sqrt_2l_1() const { return sqrt_2l_1; }
  int num_legendre() const {return n_legendre_;}

 private:
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> Tnl_;
  std::vector<double> inv_l_, sqrt_2l_1;
};
