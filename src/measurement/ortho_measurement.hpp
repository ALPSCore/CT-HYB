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
#include "../sliding_window/sliding_window.hpp"
#include "../common/legendre.hpp"
#include "../model/operator.hpp"
#include "../moves/mc_config.hpp"
#include "../common/basis.hpp"
#include "worm_meas.hpp"

void init_work_space(boost::multi_array<std::complex<double>, 3> &data, int num_flavors, int num_legendre, int num_freq);
void init_work_space(boost::multi_array<std::complex<double>, 7> &data, int num_flavors, int num_legendre, int num_freq);

/**
 * @brief Helper struct for measurement of Green's function using Legendre basis in G space
 */
template<typename SCALAR, int RANK>
struct MeasureGHelper {
  static void perform(double beta,
                      std::shared_ptr<OrthogonalBasis> p_basis,
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
  static int perform(double beta,
                      std::shared_ptr<OrthogonalBasis> p_basis,
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
  static int perform(double beta,
                      std::shared_ptr<OrthogonalBasis> p_basis,
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
template<typename SCALAR, typename SW_TYPE, int Rank>
class GOrthoBasisMeasurement : public WormMeas<SCALAR,SW_TYPE> {
 public:
  /**
   * Constructor
   *
   * @param num_flavors    the number of flavors
   * @param num_legendre   the number of legendre coefficients
   * @param num_freq       the number of bosonic frequencies
   * @param beta           inverse temperature
   */
  GOrthoBasisMeasurement(
      alps::random01 *p_rng,
      double beta,
      int num_flavors,
      std::shared_ptr<OrthogonalBasis> p_basis,
      int num_freq,
      int max_num_ops,
      double eps = 1E-5,
      int max_num_data = 1,
      std::string name="G"+boost::lexical_cast<std::string>(Rank)) :
      p_rng_(p_rng),
      str_(name),
      num_flavors_(num_flavors),
      p_basis_(p_basis),
      num_freq_(num_freq),
      beta_(beta),
      max_num_ops_(max_num_ops),
      eps_(eps),
      num_data_(0),
      max_num_data_(max_num_data) {
    init_work_space(data_, num_flavors, p_basis->dim(), num_freq);
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
  virtual void measure(
      const MonteCarloConfiguration<SCALAR> &mc_config,
      const SW_TYPE &sliding_window,
      alps::accumulators::accumulator_set &measurements);

 private:
  alps::random01 *p_rng_;
  std::string str_;
  int num_flavors_, num_freq_;
  std::shared_ptr<OrthogonalBasis> p_basis_;
  double beta_;
  int max_num_ops_;
  double eps_;
  //flavor, ..., flavor, freq_basis, freq_basis, ..., freq_basis
  boost::multi_array<std::complex<double>, 4 * Rank - 1> data_;
  int num_data_;
  int max_num_data_;//max number of data accumlated before passing data to ALPS
};