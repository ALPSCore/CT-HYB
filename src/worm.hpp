#pragma once

#include <boost/array.hpp>
#include <boost/multi_array.hpp>

#include "operator.hpp"

/**
 * < T_tau e^{-beta H} O1 (tau1) O2 (tau) ... ON(tau)>
 */
class Worm {
 public:

  virtual boost::shared_ptr<Worm> clone() const = 0;

  /** Get the number of creation and annihilation operators (not time-ordered)*/
  virtual int num_operators() const = 0;

  /** Get creation and annihilation operators (not time-ordered)*/
  virtual std::vector<psi> get_operators() const = 0;

  virtual int num_independent_times() const = 0;

  virtual double get_time(int index) const = 0;

  virtual void set_time(int index, OperatorTime new_time) const = 0;

  virtual int num_independent_flavors() const = 0;

  virtual int get_flavor(int index) const = 0;

  virtual void set_flavor(int index, int new_flavor) const = 0;

  virtual const std::vector<int> &get_time_index(int flavor_index) const = 0;

  //virtual const std::vector<int> &get_flavor_index(int time_index) const = 0;
};

inline bool is_worm_in_range(const Worm &worm, double tau_low, double tau_high) {
  assert(tau_low <= tau_high);
  const int num_times = worm.num_independent_times();
  for (int t = 0; t < num_times; ++t) {
    if (worm.get_time(t) < tau_low || worm.get_time(t) > tau_high) {
      return false;
    }
  }
  return true;
}

/**
 * Measure < N_{i_0 i_0^\prime} (tau_0) ... N_{i_{N-1} i_{N-1}^\prime} (tau_{N-1})
 * N = NumTimes
 */
template<unsigned int NumTimes>
class CorrelationWorm: public Worm {
 public:
  CorrelationWorm() : time_index_(2 * NumTimes) {
    for (int f = 0; f < 2 * NumTimes; ++f) {
      time_index_[f] = f / 2;
    }
  }

  virtual boost::shared_ptr<Worm> clone() const {
    return boost::shared_ptr<Worm>(new CorrelationWorm<NumTimes>(*this));
  }

  virtual int num_operators() const { return 2 * NumTimes; };

  virtual std::vector<psi> get_operators() const;//implemented in worm.ipp

  virtual int num_independent_times() const { return NumTimes; }

  virtual double get_time(int index) const {
    assert(index >= 0 && index < NumTimes);
    return times_[index];
  }

  virtual void set_time(int index, double new_time) const {
    assert(index >= 0 && index < NumTimes);
    times_[index] = new_time;
  }

  virtual int num_independent_flavors() const { return 2 * NumTimes; }

  virtual int get_flavor(int index) const {
    assert(index >= 0 && index < 2 * NumTimes);
    return flavors_[index];
  }

  virtual void set_flavor(int index, int new_flavor) const {
    assert(index >= 0 && index < 2 * NumTimes);
    flavors_[index] = new_flavor;
  }

  virtual const std::vector<int> &get_time_index(int flavor_index) const {
    assert(index >= 0 && index < 2 * NumTimes);
    return time_index[flavor_index];
  }

 private:
  boost::array<double, NumTimes> times_;
  boost::array<int, 2 * NumTimes> flavors_;
  std::vector<std::vector<int> > time_index_;
};
