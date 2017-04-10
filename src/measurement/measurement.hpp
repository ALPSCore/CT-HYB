#pragma once

#include <algorithm>
#include <functional>

#include <boost/multi_array.hpp>
#include <boost/range/algorithm.hpp>

#include <Eigen/Dense>

#include <alps/fastupdate/resizable_matrix.hpp>
#include <alps/mc/random01.hpp>
#include <alps/accumulators.hpp>

#include "../accumulator.hpp"
#include "../mc_config.hpp"
#include "../sliding_window/sliding_window.hpp"
#include "src/orthogonal_basis/basis.hpp"
#include "../operator.hpp"


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
   * @param Lambda   parameter for the IR basis
   * @param max_dim   the maximum dimension of the IR basis
   * @param num_matsubara  the number of Matsubara frequencies
   * @param beta           inverse temperature
   */
  GreensFunctionLegendreMeasurement(int num_flavors, double Lambda, int max_dim, double beta) :
      num_flavors_(num_flavors),
      beta_(beta),
      temperature_(1.0 / beta),
      basis_(Lambda, max_dim),
      n_meas_(0),
      g_meas_(boost::extents[num_flavors][num_flavors][basis_.dim()]) {
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
    std::vector<double> Pl_vals(basis_.dim());
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
      std::vector<double> sqrt_2l_1(basis_.dim());
      for (int il = 0; il < basis_.dim(); ++il) {
        sqrt_2l_1[il] = sqrt(2.0/basis_.norm2(il));
      }

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
          basis_.value(x, Pl_vals);
          const SCALAR coeff = -M(l, k) * bubble_sign * mc_config.sign * temperature_;
          for (int il = 0; il < basis_.dim(); ++il) {
            g_meas_[flavor_a][flavor_c][il] += coeff * sqrt_2l_1[il] * Pl_vals[il];
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
                     3> get_measured_coefficients(const Eigen::MatrixBase<Derived> &rotmat_Delta) const {
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
    for (int il = 0; il < basis_.dim(); ++il) {
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
   * Returns the number of ir coefficients
   */
  inline int get_num_ir() const { return basis_.dim(); }

  inline bool has_samples() const { return n_meas_ > 0; }

  inline int num_samples() const { return n_meas_; }

 private:
  const int num_flavors_;
  const double beta_, temperature_;
  FermionicIRBasis basis_;
  int n_meas_;
  boost::multi_array<std::complex<double>, 3> g_meas_;
};

/**
 * Helper for measuring kL and KR
 */
template<class OP, typename T>
class Window: public std::unary_function<OP, bool> {
 public:

  Window(const T itau_lower, const T itau_upper)
      : lower_(itau_lower), upper_(itau_upper) {
  }

  bool operator()(const OP &op) {
    return (op.time() >= lower_) && (op.time() < upper_);
  }
 private:
  const T lower_;
  const T upper_;
};

/**
 * Measure kL and KR for fidelity susceptibility
 */
inline unsigned measure_kLkR(const operator_container_t& operators, const double beta, const double start=0.0) {
  Window<psi,OperatorTime> is_left(OperatorTime(start,0), OperatorTime(start+ 0.5*beta,0));
  unsigned kL = std::count_if(operators.begin(), operators.end(), is_left);
  return kL*(operators.size()-kL);
}

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

/**
 * @brief Class for measurement of two-time correlation function using Legendre basis
 */
template<typename SCALAR>
class TwoTimeG2Measurement {
 public:
  /**
   * Constructor
   *
   * @param num_flavors    the number of flavors
   * @param Lambda         Lambda for bosonic IR basis
   * @param max_dim        the maximum dimension of bosonic IR basis
   * @param beta           inverse temperature
   */
  TwoTimeG2Measurement(int num_flavors, double Lambda, int max_dim, double beta) :
      num_flavors_(num_flavors),
      beta_(beta),
      basis_(Lambda, max_dim),
      data_(boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][basis_.dim()]) {
    std::fill(data_.origin(), data_.origin() + data_.num_elements(), 0.0);
  }

  /**
   * @brief Measure correlation functions with shifting two of the four operators on the interval [0,beta]
   * @param average_pert_order average perturbation order, which is used for determining the number of shifts
   */
  template<typename SlidingWindow>
  void measure(MonteCarloConfiguration<SCALAR> &mc_config,
               alps::accumulators::accumulator_set &measurements,
                   alps::random01 &random, SlidingWindow &sliding_window, int average_pert_order, const std::string &str);

 private:
  /** Measure correlation functions for a single configuration */
  void measure_impl(const std::vector<psi> &worm_ops, SCALAR weight,
                    boost::multi_array<std::complex<double>,5> &data);
  int num_flavors_;
  double beta_;
  BosonicIRBasis basis_;
  boost::multi_array<std::complex<double>, 5> data_;
};

void init_work_space(boost::multi_array<std::complex<double>, 3> &data, int num_flavors, int num_ir, int num_freq);
void init_work_space(boost::multi_array<std::complex<double>, 7> &data, int num_flavors, int num_ir, int num_freq);

/**
 * @brief Helper struct for measurement of Green's function using Legendre basis in G space
 */
template<typename SCALAR, int RANK>
struct MeasureGHelper {
  static void perform(double beta,
                      const FermionicIRBasis &basis,
                      int n_freq,
                      SCALAR sign, SCALAR weight_rat_intermediate_state,
                      const std::vector<psi> &creation_ops,
                      const std::vector<psi> &annihilation_ops,
                      const alps::fastupdate::ResizableMatrix<SCALAR> &M,
                      boost::multi_array<std::complex<double>, 4 * RANK -1> &data
  );
};

/**
 * @brief Specialization for measureing G1
 */
template<typename SCALAR>
struct MeasureGHelper<SCALAR, 1> {
  static void perform(double beta,
                      const FermionicIRBasis &basis,
                      int n_freq,
                      SCALAR sign, SCALAR weight_rat_intermediate_state,
                      const std::vector<psi> &creation_ops,
                      const std::vector<psi> &annihilation_ops,
                      const alps::fastupdate::ResizableMatrix<SCALAR> &M,
                      boost::multi_array<std::complex<double>, 3> &data
  );
};

/**
 * @brief Specialization for measureing G2
 */
template<typename SCALAR>
struct MeasureGHelper<SCALAR, 2> {
  static void perform(double beta,
                      const FermionicIRBasis &basis,
                      int n_freq,
                      SCALAR sign, SCALAR weight_rat_intermediate_state,
                      const std::vector<psi> &creation_ops,
                      const std::vector<psi> &annihilation_ops,
                      const alps::fastupdate::ResizableMatrix<SCALAR> &M,
                      boost::multi_array<std::complex<double>, 7> &data
  );
};

/**
 * @brief Class for measurement of Green's function using Legendre basis
 */
template<typename SCALAR, int Rank>
class GMeasurement {
 public:
  /**
   * Constructor
   *
   * @param num_flavors    the number of flavors
   * @param Lambda_f   Lambda for fermionic ir basis
   * @param max_dim_f   max dim of fermionic ir basis
   * @param num_freq       the number of bosonic frequencies
   * @param beta           inverse temperature
   */
  GMeasurement(int num_flavors,
               double Lambda_f,
               int max_dim_f, int num_freq, double beta, int max_num_data = 1) :
      str_("G"+boost::lexical_cast<std::string>(Rank)),
      num_flavors_(num_flavors),
      num_freq_(num_freq),
      beta_(beta),
      basis_(Lambda_f, max_dim_f),
      num_data_(0),
      max_num_data_(max_num_data) {
    init_work_space(data_, num_flavors, basis_.dim(), num_freq);
  };

  /**
   * @brief Create ALPS observable
   */
  void create_alps_observable(alps::accumulators::accumulator_set &measurements) const {
    create_observable<std::complex<double>, SimpleRealVectorObservable>(measurements, str_.c_str());
  }

  /**
   * @brief Measure Green's function via hybridization function
   */
  void measure_via_hyb(const MonteCarloConfiguration<SCALAR> &mc_config,
               alps::accumulators::accumulator_set &measurements,
               alps::random01 &random, int max_matrix_size, double eps = 1E-5);

 private:
  std::string str_;
  int num_flavors_, num_freq_;
  double beta_;
  FermionicIRBasis basis_;
  //flavor, ..., flavor, ir, ir, ..., ir
  boost::multi_array<std::complex<double>, 4 * Rank - 1> data_;
  int num_data_;
  int max_num_data_;//max number of data accumlated before passing data to ALPS
};

/**
 * @brief Class for measurement of equal-time Green's function
 */
template<typename SCALAR, int Rank>
class EqualTimeGMeasurement {
 public:
  /**
   * Constructor
   *
   * @param num_flavors    the number of flavors
   */
  EqualTimeGMeasurement(int num_flavors) :
      num_flavors_(num_flavors) {};

  /**
   * Measurement of equal-time single-particle GF
   */
  void measure_G1(MonteCarloConfiguration<SCALAR> &mc_config,
              alps::accumulators::accumulator_set &measurements,
              const std::string &str);

  /**
   * Measurement of equal-time two-particle GF
   */
  void measure_G2(MonteCarloConfiguration<SCALAR> &mc_config,
               alps::accumulators::accumulator_set &measurements,
               const std::string &str);

 private:
  int num_flavors_;
};

//#include "measurement.ipp"
