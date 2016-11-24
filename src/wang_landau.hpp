#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

/**
 * @brief Flat histogram method based on Wang-landau algorithm
 */
class FlatHistogram {
 public:
  typedef unsigned int uint;

  /**
   * @param max_val The value takes 0 - max_val.
   */
  FlatHistogram(unsigned int max_val)
      : max_val_(max_val), num_bin_(max_val + 1),
        criterion(0.8),
        min_count(std::max(10.0, 1 / ((1.0 - criterion) * (1.0 - criterion)))),
        init_log_lambda_(0.993251773010283), //std::log(2.7);
        min_log_lambda_(0.000099995000334), //std::log(1.0001);
        log_lambda_(init_log_lambda_), log_f_(num_bin_, 0),
        counter_(num_bin_, 0),
        done_(false),
        top_index_(0),
        max_index_(0),
        has_guess_(true),
        num_updates_lambda_(0),
        target_fractions_(max_val + 1, 1.0){
    max_index_ = max_val;
  }

  /**
   * @param max_val The value takes 0 - max_val.
   */
  FlatHistogram(unsigned int max_val, const std::vector<double> &target_fractions)
      : max_val_(max_val), num_bin_(max_val + 1),
        criterion(0.8),
        min_count(std::max(10.0, 1 / ((1.0 - criterion) * (1.0 - criterion)))),
        init_log_lambda_(0.993251773010283), //std::log(2.7);
        min_log_lambda_(0.000099995000334), //std::log(1.0001);
        log_lambda_(init_log_lambda_), log_f_(num_bin_, 0),
        counter_(num_bin_, 0),
        done_(false),
        top_index_(0),
        max_index_(0),
        has_guess_(true),
        num_updates_lambda_(0) {
    max_index_ = max_val;

    if (target_fractions.size() != max_val + 1) {
      throw std::runtime_error("size of target fractions is wrong");
    }
    target_fractions_ = target_fractions;
  }

  /**
   * Get the ratio of the values of the density of states D(value_new)/D(value_old)
   */
  double weight_ratio(uint value_new, uint value_old) const {
    if (value_new > max_index_) {
      return 0;
    }

    const double max_log_val = 115.1292546497023; //log(1E+50)
    const double min_log_val = -max_log_val;
    const double log_val = log_weight(value_new) - log_weight(value_old);
    if (log_val > max_log_val) {
      return 1E+50;
    }
    if (log_val < min_log_val) {
      return 1E-50;
    }
    return std::exp(log_val);
  }

  /**
   * Get the estimate of density of states
   */
  double get_histogram(uint value) const {
    if (!done_) {
      throw std::runtime_error("meas_measurement is called when done_==false.");
    }

    if (log_f_[value] < std::log(1E-10)) {
      return 0.0;
    } else {
      return std::exp(log_f_[value]);
    }
  }

  /**
   * Add a sample
   */
  void measure(uint value) {
    if (done_ || !has_guess_) return;

    check_range(value);
    if (value > max_index_)
      return;

    log_f_[value] += log_lambda_/target_fractions_[value];
    counter_[value] += 1.0/target_fractions_[value];
  }

  /**
   * Check if the histogram is flat enough: i.e., all values are almost equally visited.
   */
  bool flat_enough() const {
    const double mean = std::accumulate(counter_.begin(), counter_.end(), 0.0) / (max_index_ + 1);
    const double min = *std::min_element(counter_.begin(), counter_.end());
    if (min < min_count) {
      return false;
    }
    return min / mean > criterion;
  }

  /**
   * Update the estimate of the histogram.
   * This should be called once flat_enough() returns true
   */
  void update_lambda(bool verbose = false) {
    if (done_ || !has_guess_) return;

    ++num_updates_lambda_;
    log_lambda_ = std::max(
        -2.0 * std::log(num_updates_lambda_),
        std::max(0.5 * log_lambda_, min_log_lambda_)
    );//limited by 1/num_updates_lambda**2
    if (verbose) {
      std::cout << " new lambda = " << std::exp(log_lambda_) << std::endl;
      std::cout << " new log_lambda = " << log_lambda_ << std::endl;
    }
    rescale_f();

    if (verbose) {
      std::cout << " mean = " << std::accumulate(counter_.begin(), counter_.end(), 0.0) / (max_index_ + 1) << std::endl;
      std::cout << " max = " << *std::max_element(counter_.begin(), counter_.end()) << std::endl;
      std::cout << " min = " << *std::min_element(counter_.begin(), counter_.end()) << std::endl;
      for (int i = 0; i < max_index_ + 1; ++i)
        std::cout << " counter  " << i << " " << counter_[i] << " " << log_f_[i] << std::endl;
    }

    std::fill(counter_.begin(), counter_.end(), 0);
  }

  /**
   * Return if the estimate of histogram is done.
   */
  bool is_learning_done() {
    return done_;
  }

  /**
   * Return if it's converged
   */
  bool converged() {
    return log_lambda_ < 1.01 * min_log_lambda_;
  }

  /**
   * Synchronize log_f over MPI processes
   */
  void synchronize(const alps::mpi::communicator & comm) {
    if (is_learning_done()) {
      throw std::runtime_error("synchronize() is called after learning is done.");
    }

    std::vector<double> log_f_out(log_f_.size());
    MPI_Allreduce((void *) &log_f_[0],
                  (void *) &log_f_out[0],
                  log_f_.size(),
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);
    std::transform(log_f_out.begin(), log_f_out.end(), log_f_.begin(),
                   std::bind2nd(std::divides<double>(), 1.*comm.size())
    );
  }

  /**
   * Finish the estimate of histogram
   */
  void finish_learning(bool verbose) {
    done_ = true;

    rescale_f();
    std::fill(counter_.begin(), counter_.end(), 0);
    if (verbose) {
      std::cout << " log_f for measurement steps " << std::endl;
      for (int i = 0; i < max_index_ + 1; ++i)
        std::cout << " value " << i << " " << log_f_[i] << std::endl;
    }
  }

 private:
  const uint max_val_, num_bin_;
  std::vector<double> target_fractions_;
  double criterion;
  double min_count;
  double init_log_lambda_;
  double min_log_lambda_;

  double log_lambda_;
  std::vector<double> log_f_, log_dos_guess_;
  std::vector<double> counter_;
  uint top_index_, max_index_;
  bool done_, has_guess_;

  long num_updates_lambda_;//how many time lambda has been updated

  void check_range(uint value) const {
    if (value > max_val_) {
      throw std::runtime_error("value is not within the range.");
    }
  }

  void rescale_f() {
    const double max_log_f = *std::max_element(log_f_.begin(), log_f_.end());
    for (int i = 0; i < max_index_ + 1; ++i)
      log_f_[i] -= max_log_f;
  }

 public:
  double log_weight(uint value) const {
    if (!has_guess_)
      return 0.;

    if (value <= max_index_) {
      return -log_f_[value];
    } else {
      throw std::runtime_error("Not within range!");
    }
  }
};
