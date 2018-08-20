#pragma once

#include <complex>
#include <cmath>
#include <vector>
#include <assert.h>
#include <algorithm>

#include "boost/math/special_functions/bessel.hpp"
#include "boost/multi_array.hpp"

#include<Eigen/Dense>

#include "./thirdparty/irbasis.hpp"

#include "util.hpp"

template<typename T>
Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
convert_to_eigen_matrix(const std::vector<std::vector<T>>& array) {
  int N1 = array.size();
  int N2 = array[0].size();

  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> mat(N1, N2);

  for (int j=0; j<N2; ++j) {
    for (int i=0; i<N1; ++i) {
      mat(i,j) = array[i][j];
    }
  }

  return mat;
}

class IRbasis {
 public:
  IRbasis(double Lambda, double beta, const std::string& file_name);

  double beta() const {return beta_;}
  int dim_F() const {return basis_f_.dim();}
  int dim_B() const {return basis_b_.dim();}

  Eigen::MatrixXcd compute_Unl_F(int niw) const;
  Eigen::MatrixXcd compute_Unl_B(int niw) const;

  void compute_Utau_F(double tau, std::vector<double> &val) const;
  void compute_Utau_B(double tau, std::vector<double> &val) const;

  const std::vector<double>& bin_edges() const {return bin_edges_;}

  int get_bin_index(double tau) const {
    auto tau_bounded = mymod(tau, beta_);
    std::size_t idx = std::distance(
            bin_edges_.begin(),
            std::upper_bound(bin_edges_.begin(), bin_edges_.end(), tau_bounded)) - 1;

    return std::min(idx, bin_edges_.size() - 2);
  }

private:
    double Lambda_, beta_;
    irbasis::basis basis_f_, basis_b_;
    std::vector<double> bin_edges_;
};
