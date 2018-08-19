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

 private:
  double Lambda_, beta_;
  irbasis::basis basis_f_, basis_b_;

 public:
  double beta() const {return beta_;}
  int dim_F() const {return basis_f_.dim();}
  int dim_B() const {return basis_b_.dim();}

  Eigen::MatrixXcd compute_Unl_F(int niw) const;
  Eigen::MatrixXcd compute_Unl_B(int niw) const;

  void compute_Utau_F(double tau, std::vector<double> &val) const;
  void compute_Utau_B(double tau, std::vector<double> &val) const;
};
