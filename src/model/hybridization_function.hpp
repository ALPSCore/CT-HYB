#pragma once

#include <iostream>
#include <fstream>

#include <boost/multi_array.hpp>
#include <boost/lexical_cast.hpp>

#include <alps/hdf5/archive.hpp>

#include "../hdf5/boost_any.hpp"
#include "../common/util.hpp"
#include "operator.hpp"

template<typename SCALAR>
class HybridizationFunction {
 private:
  typedef boost::multi_array<SCALAR, 3> container_t;

 public:
  HybridizationFunction(double BETA, int n_tau, int n_flavors, const container_t &F, double eps = 1e-10) :
      BETA_(BETA),
      F_(F),
      Delta_(F),
      n_tau_(n_tau),
      n_flavors_(n_flavors),
      connected_(boost::extents[n_flavors_][n_flavors_]) {
    check_true(F_[0][0].size() == n_tau + 1);
    check_true(connected_.shape()[0] == n_flavors);
    check_true(connected_.shape()[1] == n_flavors);
    for (auto t=0; t<n_tau+1; ++t) {
      for (auto f0=0; f0 < n_flavors; ++f0) {
        for (auto f1=0; f1 < n_flavors; ++f1) {
          Delta_[f0][f1][n_tau-t] = - F_[f1][f0][t];
        }
      }
    }
    construct_connected(eps);
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

  void save_info_for_postprocessing(const std::string &filename) const {
    alps::hdf5::archive oar(filename, "a");
    oar["/F_tau"] << boost::any(F_);
    oar["/Delta_tau"] << boost::any(Delta_);
    oar.close();
  }

 private:
  double BETA_;
  int n_tau_, n_flavors_;
  container_t F_, Delta_;
  boost::multi_array<bool, 2> connected_;

  void construct_connected(double eps) {
    for (auto flavor = 0; flavor < n_flavors_; ++flavor) {
      for (auto flavor2 = 0; flavor2 < n_flavors_; ++flavor2) {
        connected_[flavor][flavor2] = false;
        for (int itau = 0; itau < n_tau_ + 1; ++itau) {
          if (std::abs(F_[flavor][flavor2][itau]) > eps) {
            connected_[flavor][flavor2] = true;
          }
        }
      }
    }
  }
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
  Delta_.resize(boost::extents[n_flavors][n_flavors][Np1_]);
  connected_.resize(boost::extents[n_flavors_][n_flavors_]);
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
        Delta_[i][j][time] = mycast<SCALAR>(std::complex<double>(real, imag));
        F_[j][i][Np1_ - time - 1] = -Delta_[i][j][time];
      }
    }
  }
  construct_connected(eps);
}
