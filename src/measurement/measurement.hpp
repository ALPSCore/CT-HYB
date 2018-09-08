#pragma once

#include <algorithm>
#include <functional>

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
#include "src/gf_basis.hpp"
#include "../operator.hpp"


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
               alps::random01 &random, int max_matrix_size, int Rank, double eps = 1E-5);

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

/**
 * @brief Class for measurement of Green's function using Legendre basis
 */
template<typename SCALAR>
class G1Measurement {
public:
    /**
     * Constructor
     *
     * @param num_flavors    the number of flavors
     */
    G1Measurement(int num_flavors, const IRbasis& basis, int max_num_data = 1) :
            str_("G1"),
            num_flavors_(num_flavors),
            data_(boost::extents[num_flavors][num_flavors][basis.dim_F()]),
            bin_data_(boost::extents[num_flavors][num_flavors][basis.bin_edges().size()-1]),
            num_data_(0),
            max_num_data_(max_num_data) {
        std::fill(data_.origin(), data_.origin() + data_.num_elements(), 0);
        std::fill(bin_data_.origin(), bin_data_.origin() + bin_data_.num_elements(), 0);
    };

    /**
     * @brief Create ALPS observable
     */
    void create_alps_observable(alps::accumulators::accumulator_set &measurements) const {
      create_observable<std::complex<double>, SimpleRealVectorObservable>(measurements, str_.c_str());
      create_observable<std::complex<double>, SimpleRealVectorObservable>(measurements, (str_ + "_bin").c_str());
      measurements << alps::accumulators::NoBinningAccumulator<std::vector<double>>(str_ + "_bin_histogram");
    }

    /**
     * @brief Measure Green's function via hybridization function
     */
    void measure_via_hyb(const MonteCarloConfiguration<SCALAR> &mc_config,
                         const IRbasis& basis,
                         alps::accumulators::accumulator_set &measurements,
                         alps::random01 &random, int max_matrix_size, double eps = 1E-5);

private:
    std::string str_;
    int num_flavors_;
    boost::multi_array<std::complex<double>, 3> data_; //flavor, flavor, IR
    boost::multi_array<std::complex<double>, 3> bin_data_; //flavor, flavor, bins
    int num_data_;
    int max_num_data_;//max number of data accumlated before passing data to ALPS
};

using matsubara_freq_point_PH = std::tuple<int,int,int>;

template<typename SCALAR>
class G2Measurement {
public:
    /**
     * Constructor
     *
     * @param num_flavors    the number of flavors
     */
    G2Measurement(int num_flavors, const IRbasis& basis, const std::vector<matsubara_freq_point_PH>& freqs, int num_freq_b, int max_num_data = 1) :
            str_("G2"),
            num_flavors_(num_flavors),
            freqs_(freqs),
            matsubara_data_(boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][freqs_.size()]),
            num_data_(0),
            max_num_data_(max_num_data) {
        std::fill(matsubara_data_.origin(), matsubara_data_.origin() + matsubara_data_.num_elements(), 0);
    };

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
                         const IRbasis& basis,
                         alps::accumulators::accumulator_set &measurements,
                         alps::random01 &random, int max_matrix_size, double eps = 1E-5);

private:
    std::string str_;
    int num_flavors_;
    std::vector<matsubara_freq_point_PH> freqs_;
    boost::multi_array<std::complex<double>, 5> matsubara_data_; //flavor, flavor, flavor, flavor, freq
    int num_data_;
    int max_num_data_;//max number of data accumlated before passing data to ALPS
};

template<typename SCALAR>
class G2IRMeasurement {
public:
    /**
     * Constructor
     *
     * @param num_flavors    the number of flavors
     */
    G2IRMeasurement(int num_flavors, const IRbasis& basis, int max_num_data = 1) :
        str_("G2"),
        num_flavors_(num_flavors),
        num_bins_(basis.num_bins_4pt()),
        data_(boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][num_bins_]),
        num_data_(0),
        max_num_data_(max_num_data) {
      std::fill(data_.origin(), data_.origin() + data_.num_elements(), 0);
    };

    /**
     * @brief Create ALPS observable
     */
    void create_alps_observable(alps::accumulators::accumulator_set &measurements) const {
      create_observable<std::complex<double>, SimpleRealVectorObservable>(measurements, (str_ + "_bin").c_str());
    }

    /**
     * @brief Measure Green's function via hybridization function
     */
    void measure_via_hyb(const MonteCarloConfiguration<SCALAR> &mc_config,
                         const IRbasis& basis,
                         alps::accumulators::accumulator_set &measurements,
                         alps::random01 &random, int max_matrix_size, double eps = 1E-5);

private:
    std::string str_;
    int num_flavors_;
    int num_bins_;
    boost::multi_array<std::complex<double>, 5> data_; //flavor, flavor, flavor, flavor, bins
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

std::vector<matsubara_freq_point_PH> read_matsubara_points(const std::string& file);

//#include "measurement.ipp"