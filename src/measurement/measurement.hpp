#pragma once

#include <algorithm>
#include <functional>
#include <unordered_map>

#include <boost/multi_array.hpp>
#include <boost/range/algorithm.hpp>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <alps/fastupdate/resizable_matrix.hpp>
#include <alps/mc/random01.hpp>
#include <alps/accumulators.hpp>

#include "../accumulator.hpp"
#include "../mc_config.hpp"
#include "../sliding_window/sliding_window.hpp"
#include "../legendre.hpp"
#include "../operator.hpp"
#include "../hash.hpp"

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
   * @param num_legendre   the number of legendre coefficients
   * @param beta           inverse temperature
   */
  TwoTimeG2Measurement(int num_flavors, int num_legendre, double beta) :
      num_flavors_(num_flavors),
      beta_(beta),
      legendre_trans_(1, num_legendre),
      data_(boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][num_legendre]) {
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
  LegendreTransformer legendre_trans_;
  boost::multi_array<std::complex<double>, 5> data_;
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
 * @brief Perform reconnections of hybridization lines
 *
 * Rank: number of aux lines to be added in constructing an intermediate matrix
 */
template<typename SCALAR>
class Reconnections {
 public:
  Reconnections(const MonteCarloConfiguration<SCALAR> &mc_config,
               alps::random01 &random, int max_matrix_size, int Rank, double eps);

  const alps::fastupdate::ResizableMatrix<SCALAR>& M() const {
    return M_;
  }

  const std::vector<psi>& creation_ops() const {
    return creation_ops_;
  }

  const std::vector<psi>& annihilation_ops() const {
    return annihilation_ops_;
  }

  SCALAR weight_rat_intermediate_state() const {
    return weight_rat_intermediate_state_;
  }

private:
  alps::fastupdate::ResizableMatrix<SCALAR> M_;
  SCALAR weight_rat_intermediate_state_;
  std::vector<psi> creation_ops_;
  std::vector<psi> annihilation_ops_;
};

using matsubara_freq_point_PH = std::tuple<int,int,int>;

void make_two_freqs_list(
    const std::vector<matsubara_freq_point_PH>& freqs,
    std::vector<std::pair<int,int>>& two_freqs_vec,
    std::unordered_map<std::pair<int,int>, int>& two_freqs_map);

template<typename SCALAR>
class G2Measurement {
public:
    /**
     * Constructor
     *
     * @param num_flavors    the number of flavors
     */
    G2Measurement(int num_flavors, double beta, const std::vector<matsubara_freq_point_PH>& freqs);


    /**
     * @brief Create ALPS observable
     */
    void create_alps_observable(alps::accumulators::accumulator_set &measurements) const {
        create_observable<std::complex<double>, SimpleRealVectorObservable>(measurements, (str_ + "_matsubara").c_str());
    }

    /**
     * @brief Measure Green's function via hybridization function
     */
    void measure_via_hyb(const MonteCarloConfiguration<SCALAR> &mc_config,
                         alps::random01 &random, int max_matrix_size, double eps = 1E-5);

    void finalize(const std::string& output_file);

private:
    std::string str_;
    int num_flavors_;
    double beta_;
    std::vector<matsubara_freq_point_PH> freqs_;
    boost::multi_array<std::complex<double>, 5> matsubara_data_; //flavor, flavor, flavor, flavor, freq
    int num_data_;
    int max_num_data_;//max number of data accumlated before passing data to ALPS
    std::vector<std::pair<int,int>> two_freqs_vec_;
    std::unordered_map<std::pair<int,int>, int> two_freqs_map_;
};


inline
matsubara_freq_point_PH
from_H_to_F(const matsubara_freq_point_PH& freq_PH) {
  int freq_f1 = std::get<0>(freq_PH);
  int freq_f2 = std::get<1>(freq_PH);
  int freq_b = std::get<2>(freq_PH);
  return matsubara_freq_point_PH(freq_f2+freq_b, freq_f2, freq_f1-freq_f2);
}

inline
matsubara_freq_point_PH
from_H_to_F(int freq_f1, int freq_f2, int freq_b) {
  return from_H_to_F(matsubara_freq_point_PH(freq_f1, freq_f2, freq_b));
}


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

std::vector<matsubara_freq_point_PH> read_matsubara_points(const std::string& file);

//#include "measurement.ipp"
