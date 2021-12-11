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
#include "../sliding_window/sliding_window.hpp"
#include "../common/legendre.hpp"
#include "../model/operator.hpp"
#include "../moves/mc_config.hpp"
#include "../hash.hpp"
#include "worm_meas.hpp"

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

template<typename SCALAR, typename SW_TYPE>
class G2SparseMeasurement : public WormMeas<SCALAR,SW_TYPE> {
public:
    /**
     * Constructor
     *
     * @param num_flavors    the number of flavors
     */
    G2SparseMeasurement(alps::random01 *p_rng, double beta, int num_flavors,
      const std::vector<matsubara_freq_point_PH>& freqs,
      int max_matrix_size, double eps = 1E-5);

    /**
     * @brief Create ALPS observable
     */
    virtual void create_alps_observable(alps::accumulators::accumulator_set &measurements) const {
        create_observable<std::complex<double>, SimpleRealVectorObservable>(measurements, "G2H");
    }

    /**
     * @brief Measure Green's function via hybridization function
     */
    virtual void measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements);

    virtual void save_results(const std::string& filename, const alps::mpi::communicator &comm) const;

private:
    alps::random01 *p_rng_;
    int num_flavors_;
    double beta_;
    std::vector<matsubara_freq_point_PH> freqs_;
    int max_matrix_size_;
    double eps_;
    boost::multi_array<std::complex<double>, 5> matsubara_data_; //flavor, flavor, flavor, flavor, freq
    //int num_data_;
    //int max_num_data_;//max number of data accumlated before passing data to ALPS
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

std::vector<matsubara_freq_point_PH> read_matsubara_points(const std::string& file);