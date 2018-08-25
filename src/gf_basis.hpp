#pragma once

#include <complex>
#include <cmath>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <sstream>
#include <unordered_map>

#include <boost/multi_array.hpp>
#include <boost/functional/hash.hpp>

#include<Eigen/Dense>
#include <alps/params.hpp>
#include <alps/hdf5.hpp>
#include <alps/numeric/tensors.hpp>

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

namespace detail {
    class Hash
    {
    public:
        std::size_t operator()(const std::array<int,6>& oid) const
        {
          return boost::hash_range(oid.begin(), oid.end());
        }
    };
}

class IRbasis {
 public:
  //IRbasis(double Lambda, double beta, const std::string& file_name, const std::string& file_name_4pt);
  IRbasis(const alps::params& params);

  double beta() const {return beta_;}
  int dim_F() const {return dim_F_;}
  int dim_B() const {return dim_B_;}
  double sl_F(int l) const {return basis_f_.sl(l);}

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

  int get_bin_index(double t1, double t2, double t3, double t4) const {
      return get_bin_position_4pt(get_index_4pt(t1, t2, t3, t4));
  }

  double sum_inverse_bin_volume_4pt() const {
    return norm_const_4pt_;
  }

  double bin_volume_4pt(int position) const {
    return bin_volume_4pt_[position];
  }

  int get_bin_position_4pt(const std::array<int,6>& index) const {
    return bin_index_map_4pt_.at(index);
  }

  std::array<int,6>
  get_index_4pt(double t1, double t2, double t3, double t4) const {
    std::array<double,4> taus{t1, t2, t3, t4};
    std::array<int,6> index;
    int k = 0;
    for (int i = 0; i < 4; ++i) {
      for (int j = i+1; j < 4; ++j) {
        index[k] = get_bin_index(taus[i] - taus[j]);
        ++ k;
      }
    }
    return index;
  }

  int num_bins_4pt() const {
    return bin_volume_4pt_.size();
  }

  const std::array<double,3>& bin_centroid_4pt(int bin_index) const {
    return bin_centroid_4pt_[bin_index];
  };

  void check() const;

private:
    double Lambda_, beta_;
    irbasis::basis basis_f_, basis_b_;
    int dim_F_, dim_B_;
    std::vector<double> bin_edges_;

    // 4pt
    int num_bins_4pt_;
    std::vector<double> bin_volume_4pt_;
    std::vector<std::array<int,6>> bin_index_4pt_;
    std::vector<std::array<double,3>> bin_centroid_4pt_;
    std::unordered_map<std::array<int,6>,int,detail::Hash> bin_index_map_4pt_;
    double norm_const_4pt_;
};
