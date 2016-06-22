#pragma once

#include <algorithm>
#include <functional>

#include <boost/multi_array.hpp>

#include <Eigen/Dense>

#include <alps/fastupdate/resizable_matrix.hpp>

#include "legendre.hpp"
#include "operator.hpp"


/**
 * @brief Class for measurement of single-particle Green's function using Legendre basis
 */
template<typename SCALAR>
class GreensFunctionLegendreMeasurement {
 public:
  /**
   * Constructor
   *
   * @param num_flavors    the number of flavors
   * @param num_legendre   the number of legendre coefficients
   * @param num_matsubara  the number of Matsubara frequencies
   * @param beta           inverse temperature
   */
  GreensFunctionLegendreMeasurement(int num_flavors, int num_legendre, int num_matsubra, double beta) :
      num_flavors_(num_flavors),
      num_legendre_(num_legendre),
      num_matsubara_(num_matsubra),
      beta_(beta),
      temperature_(1.0 / beta),
      legendre_trans_(num_matsubara_, num_legendre_),
      n_meas_(0),
      g_meas_(boost::extents[num_flavors][num_flavors][num_legendre]) {
    std::fill(g_meas_.origin(), g_meas_.origin() + g_meas_.num_elements(), 0.0);
  }

  /**
   * Measure Green's function
   *
   * @param M the inverse matrix for hybridization function
   * @param operators a set of annihilation and creation operators
   */
  void measure(const MonteCarloConfiguration<SCALAR> &mc_config) {
    //Work array for P[x_l]
    std::vector<double> Pl_vals(num_legendre_);
    ++n_meas_;
    std::vector<psi>::const_iterator it1, it2;

    if (mc_config.pert_order() == 0) {
      return;
    }

    for (int block = 0; block < mc_config.M.num_blocks(); ++block) {
      const std::vector<psi> &creation_operators = mc_config.M.get_cdagg_ops(block);
      const std::vector<psi> &annihilation_operators = mc_config.M.get_c_ops(block);

      const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> &M =
          mc_config.M.compute_inverse_matrix(block);

      //const Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic>& G =
      //mc_config.M.compute_G_matrix(block);

      for (int k = 0; k < M.rows(); k++) {
        (k == 0 ? it1 = annihilation_operators.begin() : it1++);
        for (int l = 0; l < M.cols(); l++) {
          (l == 0 ? it2 = creation_operators.begin() : it2++);
          double argument = it1->time() - it2->time();
          double bubble_sign = 1;
          if (argument > 0) {
            bubble_sign = 1;
          } else {
            bubble_sign = -1;
            argument += beta_;
          }
          assert(-0.01 < argument && argument < beta_ + 0.01);

          const int flavor_a = it1->flavor();
          const int flavor_c = it2->flavor();
          const double x = 2 * argument * temperature_ - 1.0;
          legendre_trans_.compute_legendre(x, Pl_vals);
          const SCALAR coeff = -M(l, k) * bubble_sign * mc_config.sign * temperature_;
          for (int il = 0; il < num_legendre_; ++il) {
            g_meas_[flavor_a][flavor_c][il] += coeff * legendre_trans_.get_sqrt_2l_1()[il] * Pl_vals[il];
          }
        }
      }
    }
  }

  /**
   * Return the measured data in the original basis.
   *
   * @param rotmat_Delta   rotation matrix for Delta, which defines the single-particle basis for the perturbation expansion
   */
  template<typename Derived>
  boost::multi_array<std::complex<double>,
                     3> get_measured_legendre_coefficients(const Eigen::MatrixBase<Derived> &rotmat_Delta) const {
    if (n_meas_ == 0) {
      throw std::runtime_error("Error: n_meas_=0");
    }

    //Divide by the number of measurements
    boost::multi_array<std::complex<double>, 3> result(g_meas_);
    std::transform(result.origin(), result.origin() + result.num_elements(), result.origin(),
                   std::bind2nd(std::divides<std::complex<double> >(), 1. * n_meas_)
    );

    //Transform to the original basis
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> gf_tmp(num_flavors_, num_flavors_);
    for (int il = 0; il < num_legendre_; ++il) {
      for (int flavor = 0; flavor < num_flavors_; ++flavor) {
        for (int flavor2 = 0; flavor2 < num_flavors_; ++flavor2) {
          gf_tmp(flavor, flavor2) = result[flavor][flavor2][il];
        }
      }
      gf_tmp = rotmat_Delta * gf_tmp * rotmat_Delta.adjoint();
      for (int flavor = 0; flavor < num_flavors_; ++flavor) {
        for (int flavor2 = 0; flavor2 < num_flavors_; ++flavor2) {
          result[flavor][flavor2][il] = gf_tmp(flavor, flavor2);
        }
      }
    }

    return result;
  }

  /**
   * Clear all the data measured
   */
  void reset() {
    n_meas_ = 0;
    std::fill(g_meas_.origin(), g_meas_.origin() + g_meas_.num_elements(), 0.0);
  }

  /**
   * Returns the number of legendre coefficients
   */
  inline int get_num_legendre() const { return num_legendre_; }

  inline bool has_samples() const { return n_meas_ > 0; }

  inline int num_samples() const { return n_meas_; }

 private:
  const int num_flavors_, num_legendre_, num_matsubara_;
  const double beta_, temperature_;
  LegendreTransformer legendre_trans_;
  int n_meas_;
  boost::multi_array<std::complex<double>, 3> g_meas_;
};

/**
 * Helper for measuring kL and KR
 */
template<class OP>
class Window: public std::unary_function<OP, bool> {
 public:

  Window(const double itau_lower, const double itau_upper)
      : lower_(itau_lower), upper_(itau_upper) {
  }

  bool operator()(const OP &op) {
    return (op.time() >= lower_) && (op.time() < upper_);
  }
 private:
  const double lower_;
  const double upper_;
};

/**
 * Measure kL and KR for fidelity susceptibility
 */
/*
inline unsigned measure_kLkR(const operator_container_t& operators, const double beta, const double start=0.0) {
  Window<psi> is_left(start, start+ 0.5*beta);
  unsigned kL = std::count_if(operators.begin(), operators.end(), is_left);
  return kL*(operators.size()-kL);
}
 */

/**
 * Measure acceptance rate
 */
class AcceptanceRateMeasurement {
 public:
  AcceptanceRateMeasurement() :
      num_samples_(0.0), num_accepted_(0.0) { };

  void accepted() {
    ++num_samples_;
    ++num_accepted_;
  }

  void rejected() {
    ++num_samples_;
  }

  bool has_samples() const {
    return num_samples_ > 0.0;
  }

  void reset() {
    num_samples_ = num_accepted_ = 0.0;
  }

  double compute_acceptance_rate() const {
    if (num_samples_ == 0.0) {
      throw std::runtime_error("Error in compute_acceptance_rate! There is no sample!");
    }
    return num_accepted_ / num_samples_;
  }

 private:
  double num_samples_, num_accepted_;
};
