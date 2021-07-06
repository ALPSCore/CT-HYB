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

  HybridizationFunction(double beta, const std::string &input_file, int n_tau, int n_flavors, double eps = 1e-10);

  int num_flavors() const { return n_flavors_; }

  SCALAR operator()(double t, int flavor1, int flavor2) const {
    auto sign = 1.0;
    while (t < 0) {
      t += BETA_;
      sign *= -1;
    }
    while (t > BETA_) {
      t -= BETA_;
      sign *= -1;
    }

    double n = (t / BETA_) * n_tau_;
    int n_lower = (int) n;
    return sign * (
        F_[flavor1][flavor2][n_lower] +
        (n - n_lower) * (F_[flavor1][flavor2][n_lower + 1] - F_[flavor1][flavor2][n_lower])
        );
  }

  SCALAR operator()(const psi &c_op, const psi &cdagger_op) const {
    double t = c_op.time() - cdagger_op.time();
    return this->operator()(t, c_op.flavor(), cdagger_op.flavor());
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

/*
 * Read hybridization function from a text file
 */
template <typename SCALAR>
HybridizationFunction<SCALAR>::HybridizationFunction(
  double beta, const std::string &input_file, int n_tau, int n_flavors, double eps) :
  BETA_(beta), n_tau_(n_tau), n_flavors_(n_flavors)
{
  int Np1_ = n_tau + 1;
  F_.resize(boost::extents[n_flavors][n_flavors][Np1_]);
  // read hybridization function from input file with FLAVORS+1 colums \tau, G_1_up, G_1_down, G_2_up ..., G_SITES_down)
  std::ifstream infile_f(input_file.c_str());
  if (!infile_f.is_open())
  {
    std::cerr << "Input file for F cannot be opened!" << std::endl;
    exit(1);
  }

  double real, imag;
  int dummy_it, dummy_i, dummy_j;

  for (int time = 0; time < Np1_; time++)
  {
    for (int i = 0; i < n_flavors; i++)
    {
      for (int j = 0; j < n_flavors; j++)
      {
        infile_f >> dummy_it >> dummy_i >> dummy_j >> real >> imag;
        if (dummy_it != time)
        {
          throw std::runtime_error("Format of " + input_file + 
                                   " is wrong. The value at the first colum should be " +
                                   boost::lexical_cast<std::string>(time) + "Error at line " +
                                   boost::lexical_cast<std::string>(time + 1) + ".");
        }
        if (dummy_i != i)
        {
          throw std::runtime_error("Format of " + input_file +
                                   " is wrong. The value at the second colum should be " +
                                   boost::lexical_cast<std::string>(i) + "Error at line " +
                                   boost::lexical_cast<std::string>(time + 1) + ".");
        }
        if (dummy_j != j)
        {
          throw std::runtime_error("Format of " + input_file +
                                   " is wrong. The value at the third colum should be " +
                                   boost::lexical_cast<std::string>(j) + "Error at line " +
                                   boost::lexical_cast<std::string>(time + 1) + ".");
        }
        //F_ij(tau) = - Delta_ji (beta - tau)
        F_[j][i][Np1_ - time - 1] = -mycast<SCALAR>(std::complex<double>(real, imag));
      }
    }
  }
}