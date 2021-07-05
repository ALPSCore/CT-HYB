#pragma once

#include <boost/multi_array.hpp>

#include "operator.hpp"

template<typename SCALAR>
class HybridizationFunction {
 private:
  typedef boost::multi_array<SCALAR, 3> container_t;

 public:
  HybridizationFunction(double BETA, int n_tau, int n_flavors, const container_t &F, double eps = 1e-10) :
      BETA_(BETA),
      F_(F),
      n_tau_(n_tau),
      n_flavors_(n_flavors),
      connected_(boost::extents[n_flavors_][n_flavors_]) {
    check_true(F_[0][0].size() == n_tau + 1);
    for (int flavor = 0; flavor < n_flavors; ++flavor) {
      for (int flavor2 = 0; flavor2 < n_flavors; ++flavor2) {
        connected_[flavor][flavor2] = false;
        for (int itau = 0; itau < n_tau + 1; ++itau) {
          if (std::abs(F[flavor][flavor2][itau]) > eps) {
            connected_[flavor][flavor2] = true;
          }
        }
      }
    }
  }

  int num_flavors() const { return n_flavors_; }

  SCALAR operator()(const psi &c_op, const psi &cdagger_op) const {
    double sign = 1;
    double t = c_op.time() - cdagger_op.time();
    if (t < 0) {
      t += BETA_;
      sign = -1;
    }

    double n = t / BETA_ * n_tau_;
    int n_lower = (int) n;
    const SCALAR *pF = &F_[c_op.flavor()][cdagger_op.flavor()][0];
    return sign * (pF[n_lower] + (n - n_lower) * (pF[n_lower + 1] - pF[n_lower]));
  }

  bool is_connected(int flavor1, int flavor2) const {
    return connected_[flavor1][flavor2];
  }

 private:
  double BETA_;
  int n_tau_, n_flavors_;
  container_t F_;
  boost::multi_array<bool, 2> connected_;
};