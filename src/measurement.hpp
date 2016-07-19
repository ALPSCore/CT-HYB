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
class N2CorrelationFunctionMeasurement {
 public:
  /**
   * Constructor
   *
   * @param num_flavors    the number of flavors
   * @param num_legendre   the number of legendre coefficients
   * @param beta           inverse temperature
   */
  N2CorrelationFunctionMeasurement(int num_flavors, int num_legendre, double beta) :
      num_flavors_(num_flavors),
      beta_(beta),
      legendre_trans_(1, num_legendre),
      data_(boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][num_legendre]) {
    std::fill(data_.origin(), data_.origin() + data_.num_elements(), 0.0);
  }

  template<typename SlidingWindow>
  void measure_new(MonteCarloConfiguration<SCALAR> &mc_config,
               alps::accumulators::accumulator_set &measurements,
                   alps::random01 &random, SlidingWindow &sliding_window, const std::string &str) {
    namespace bll = boost::lambda;
    typedef typename ExtendedScalar<SCALAR>::value_type EXTENDED_SCALAR;
    typedef operator_container_t::iterator Iterator;
    if (mc_config.current_config_space() != "N2_correlation") {
      return;
    }

    //Remove the left-hand-side operators
    operator_container_t ops(mc_config.operators);
    const std::vector<psi> worm_ops_original = mc_config.p_worm->get_operators();
    safe_erase(ops, worm_ops_original);

    //Generate times of the left hand operator pair c^dagger c
    //Make sure there is no duplicate
    const int num_times = 10;
    std::set<double> new_times;
    new_times.insert(worm_ops_original[0].time().time());
    if (new_times.size() < num_times) {
      std::set<double> duplicate_check;
      for (operator_container_t::iterator it = mc_config.operators.begin(); it != mc_config.operators.end(); ++it) {
        duplicate_check.insert(it->time().time());
      }
      while(new_times.size() < num_times) {
        double t = open_random(random, 0.0, beta_);
        if (duplicate_check.find(t) == duplicate_check.end()) {
          new_times.insert(t);
          duplicate_check.insert(t);
        }
      }
      assert(new_times.size() == num_times);
    }
    
    //remember the current status of the sliding window
    const typename SlidingWindow::state_t state = sliding_window.get_state();
    const int n_win = sliding_window.get_n_window();

    std::vector<EXTENDED_REAL> trace_bound(sliding_window.get_num_brakets());

    //configurations whose trace is smaller than this cutoff are ignored.
    const EXTENDED_REAL trace_cutoff = EXTENDED_REAL(1.0E-30)*myabs(mc_config.trace);

    double norm = 0.0;
    std::fill(data_.origin(), data_.origin() + data_.num_elements(), 0.0);

    //compute Monte Carlo weights of configurations with new time
    // sweep = 0: configuration with new time for the left-hand operator pair
    // sweep = 1: configuration with new time and new flavors for all worm operators
    for (int sweep = 0; sweep < 2; ++sweep) {
      std::vector<psi> worm_ops = worm_ops_original;
      if (sweep == 1) {//change flavors of all worm operators
        for (int iop = 0; iop < worm_ops.size(); ++iop) {
          worm_ops[iop].set_flavor(static_cast<int>(random()*num_flavors_));
        }
      }

      //insert worm operators which are not shifted in time
      for (int iop = 2; iop < worm_ops.size(); ++iop) {
        safe_insert(ops, worm_ops[iop]);
      }

      //reset the window and move to the right most position
      sliding_window.set_window_size(std::max(num_times, n_win), ops, 0, ITIME_LEFT);

      for (std::set<double>::iterator it = new_times.begin(); it != new_times.end(); ++it) {
        //move the window so that it contains the time
        while (*it > sliding_window.get_tau_high()) {
          sliding_window.move_window_to_next_position(ops);
        }
        assert(*it <= sliding_window.get_tau_high());
        assert(*it >= sliding_window.get_tau_low());
  
        worm_ops[0].set_time(OperatorTime(*it, +1));
        worm_ops[1].set_time(OperatorTime(*it,  0));
        safe_insert(ops, worm_ops[0]);
        safe_insert(ops, worm_ops[1]);
  
        sliding_window.compute_trace_bound(ops, trace_bound);
        std::pair<bool, EXTENDED_SCALAR> r = sliding_window.lazy_eval_trace(ops, EXTENDED_REAL(0.0), trace_bound);
        if (myabs(r.second) > trace_cutoff) {
          const SCALAR weight = convert_to_scalar(r.second/mc_config.trace);
          measure_impl(worm_ops, mc_config.sign*weight, data_);
          norm += std::abs(weight);
        }
  
        safe_erase(ops, worm_ops[0]);
        safe_erase(ops, worm_ops[1]);
      }
      
      //remove worm operators which are not shifted in time
      for (int iop = 2; iop < worm_ops.size(); ++iop) {
        safe_erase(ops, worm_ops[iop]);
      }
    }

    //normalize the data
    std::transform(data_.origin(), data_.origin() + data_.num_elements(), data_.origin(), std::bind2nd(std::divides<std::complex<double> >(), norm));

    //pass the data to ALPS libraries
    measure_simple_vector_observable<std::complex<double> >(measurements, str.c_str(), to_std_vector(data_));

    //restore the status of the sliding window
    //sliding_window.set_window_size(1, mc_config.operators, 0, ITIME_LEFT);//reset brakets
    sliding_window.restore_state(mc_config.operators, state);
  }

  void measure_impl(const std::vector<psi> &worm_ops, SCALAR weight,
                    boost::multi_array<std::complex<double>,5> &data) {
                    //alps::accumulators::accumulator_set &measurements, const std::string &str) {
    boost::array<int, 4> flavors;
    if (worm_ops[0].time().time() != worm_ops[1].time().time()) {
      throw std::runtime_error("time worm_ops0 != time worm_ops1");
    }
    if (worm_ops[2].time().time() != worm_ops[3].time().time()) {
      throw std::runtime_error("time worm_ops2 != time worm_ops3");
    }
    double tdiff = worm_ops[0].time().time() - worm_ops[2].time().time();
    if (tdiff < 0.0) {
      tdiff += beta_;
    }
    for (int f = 0; f < 4; ++f) {
      flavors[f] = worm_ops[f].flavor();
    }

    const int num_legendre = legendre_trans_.num_legendre();
    std::vector<double> Pl_vals(num_legendre);

    legendre_trans_.compute_legendre(2 * tdiff / beta_ - 1.0, Pl_vals);
    for (int il = 0; il < num_legendre; ++il) {
      data
      [flavors[0]]
      [flavors[1]]
      [flavors[2]]
      [flavors[3]][il] += weight * legendre_trans_.get_sqrt_2l_1()[il] * Pl_vals[il];
    }
  }

  void measure(const MonteCarloConfiguration<SCALAR> &mc_config,
               alps::accumulators::accumulator_set &measurements, const std::string &str) {
    if (mc_config.current_config_space() != "N2_correlation") {
      return ;
    }

    boost::array<int, 4> flavors;
    double tdiff = mc_config.p_worm->get_time(0) - mc_config.p_worm->get_time(1);
    if (tdiff >= 0.0) {
      for (int f = 0; f < 4; ++f) {
        flavors[f] = mc_config.p_worm->get_flavor(f);
      }
    } else {
      tdiff *= -1.0;
      flavors[0] = mc_config.p_worm->get_flavor(2);
      flavors[1] = mc_config.p_worm->get_flavor(3);
      flavors[2] = mc_config.p_worm->get_flavor(0);
      flavors[3] = mc_config.p_worm->get_flavor(1);
    }

    const int num_legendre =  legendre_trans_.num_legendre();
    std::vector<double> Pl_vals(num_legendre);

    legendre_trans_.compute_legendre(2 * tdiff / beta_ - 1.0, Pl_vals);
    for (int il = 0; il < num_legendre; ++il) {
      data_
      [flavors[0]]
      [flavors[1]]
      [flavors[2]]
      [flavors[3]][il] += 0.5 * mc_config.sign * legendre_trans_.get_sqrt_2l_1()[il] * Pl_vals[il];
    }

    legendre_trans_.compute_legendre(2 * (beta_ - tdiff) /beta_ - 1.0, Pl_vals);
    for (int il = 0; il < num_legendre; ++il) {
      data_
      [flavors[2]]
      [flavors[3]]
      [flavors[0]]
      [flavors[1]][il] += 0.5 * mc_config.sign * legendre_trans_.get_sqrt_2l_1()[il] * Pl_vals[il];
    }

    //measurements[str.c_str()] << to_std_vector(data_);
    measure_simple_vector_observable<std::complex<double> >(measurements, str.c_str(), to_std_vector(data_));
    std::fill(data_.origin(), data_.origin() + data_.num_elements(), 0.0);
  }

 private:
  int num_flavors_;
  double beta_;
  LegendreTransformer legendre_trans_;
  boost::multi_array<std::complex<double>, 5> data_;
};
