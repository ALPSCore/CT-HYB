#pragma once

#include <boost/array.hpp>
#include <boost/multi_array.hpp>
#include <boost/operators.hpp>

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

  /** Number of independent time indices*/
  virtual int num_independent_times() const = 0;

  /** Return a time index */
  virtual double get_time(int index) const = 0;

  /** Set a new value to a time index */
  virtual void set_time(int index, double new_time) = 0;

  /** Number of independent flavor indices */
  virtual int num_independent_flavors() const = 0;

  /** Return a flavor index */
  virtual int get_flavor(int index) const = 0;

  /** Set a new value to a flavor index */
  virtual void set_flavor(int index, int new_flavor) = 0;

  /** Return all time indices corresponding to a given flavor index */
  virtual const std::vector<int> &get_time_index(int flavor_index) const = 0;

  /** Return the name of the worm instance */
  virtual const std::string& get_name() const = 0;
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

inline bool is_worm_diagonal_in_flavor(const Worm &worm) {
  const int num_f = worm.num_independent_flavors();
  const int flavor0 = worm.get_flavor(0);
  for (int f = 1; f < num_f; ++f) {
    if (worm.get_flavor(f) != flavor0) {
      return false;
    }
  }
  return true;
}

inline bool operator==(const Worm &worm1, const Worm &worm2) {
  if (typeid(worm1) != typeid(worm2)) {
    return false;
  }
  const int num_times = worm1.num_independent_times();
  const int num_flavors = worm1.num_independent_flavors();
  bool flag = true;
  for (int t = 0; t < num_times; ++t) {
    flag = flag && worm1.get_time(t) == worm2.get_time(t);
  }
  for (int f = 0; f < num_flavors; ++f) {
    flag = flag && worm1.get_flavor(f) == worm2.get_flavor(f);
  }
  return flag;
}

inline bool operator!=(const Worm &worm1, const Worm &worm2) {
  return !(worm1 == worm2);
}

/**
 * Measure < N_{i_0 i_1} (tau_0) ... N_{i_{2M-2} i_{2M-1}} (tau_{M-1})>,
 *  where N_{ij} = c^dagger_i c_j and M = NumTimes.
 * For M=1, we measure the single-particle density matrix.
 */
template<unsigned int NumTimes>
class CorrelationWorm: public Worm, private boost::equality_comparable<CorrelationWorm<NumTimes> > {
 public:
  CorrelationWorm(const std::string &name) : name_(name), time_index_(2 * NumTimes) {
    for (int f = 0; f < 2 * NumTimes; ++f) {
      time_index_[f].push_back(f / 2);
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

  virtual void set_time(int index, double new_time) {
    assert(index >= 0 && index < NumTimes);
    times_[index] = new_time;
  }

  virtual int num_independent_flavors() const { return 2 * NumTimes; }

  virtual int get_flavor(int index) const {
    assert(index >= 0 && index < 2 * NumTimes);
    return flavors_[index];
  }

  virtual void set_flavor(int index, int new_flavor) {
    assert(index >= 0 && index < 2 * NumTimes);
    flavors_[index] = new_flavor;
  }

  virtual const std::vector<int> &get_time_index(int flavor_index) const {
    assert(flavor_index >= 0 && flavor_index < 2 * NumTimes);
    return time_index_[flavor_index];
  }

  virtual bool operator==(const CorrelationWorm<NumTimes> &other_worm) const {
    return (times_ == other_worm.times_ && flavors_ == other_worm.flavors_);
  }

  const std::string& get_name() const {
    return name_;
  }

 private:
  std::string name_;
  boost::array<double, NumTimes> times_;
  boost::array<int, 2 * NumTimes> flavors_;
  std::vector<std::vector<int> > time_index_;
};

/**
 * Measure Green's function <c c^dagger, ..., c c^dagger>
 * Rank = 1: single-particle Green's function
 * Rank = 2: two-particle Green's function
 *
 */
template<unsigned int Rank>
class GWorm: public Worm, private boost::equality_comparable<GWorm<Rank> > {
 public:
  GWorm(const std::string &name) : name_(name), time_index_(2*Rank) {
    for (int f = 0; f < 2 * Rank; ++f) {
      time_index_[f].push_back(f);
    }
  }

  virtual boost::shared_ptr<Worm> clone() const {
    return boost::shared_ptr<Worm>(new GWorm<Rank>(*this));
  }

  virtual int num_operators() const { return 2 * Rank; };

  virtual std::vector<psi> get_operators() const;//implemented in worm.ipp

  virtual int num_independent_times() const { return 2 * Rank; }

  virtual double get_time(int index) const {
    assert(index >= 0 && index < 2 * Rank);
    return times_[index];
  }

  virtual void set_time(int index, double new_time) {
    assert(index >= 0 && index < 2 * Rank);
    times_[index] = new_time;
  }

  virtual int num_independent_flavors() const { return 2 * Rank; }

  virtual int get_flavor(int index) const {
    assert(index >= 0 && index < 2 * Rank);
    return flavors_[index];
  }

  virtual void set_flavor(int index, int new_flavor) {
    assert(index >= 0 && index < 2 * Rank);
    flavors_[index] = new_flavor;
  }

  virtual const std::vector<int> &get_time_index(int flavor_index) const {
    assert(flavor_index >= 0 && flavor_index < 2 * Rank);
    return time_index_[flavor_index];
  }

  virtual bool operator==(const GWorm<Rank> &other_worm) const {
    return (times_ == other_worm.times_ && flavors_ == other_worm.flavors_);
  }

  const std::string& get_name() const {
    return name_;
  }

 private:
  std::string name_;
  boost::array<double, 2 * Rank> times_;
  boost::array<int, 2 * Rank> flavors_;
  std::vector<std::vector<int> > time_index_;
};

/**
 * Measure equal-time Green's function <c^dagger c , ..., c^dagger c>
 * Rank = 1: single-particle Green's function
 * Rank = 2: two-particle Green's function
 *
 */
template<unsigned int Rank>
class EqualTimeGWorm: public Worm, private boost::equality_comparable<EqualTimeGWorm<Rank> > {
 public:
  EqualTimeGWorm(const std::string &name) : name_(name) {
    time_index_.push_back(0);
  }

  virtual boost::shared_ptr<Worm> clone() const {
    return boost::shared_ptr<Worm>(new EqualTimeGWorm<Rank>(*this));
  }

  virtual int num_operators() const { return 2 * Rank; };

  virtual std::vector<psi> get_operators() const;//implemented in worm.ipp

  virtual int num_independent_times() const { return 1; }

  virtual double get_time(int index) const {
    assert(index == 0);
    return time_;
  }

  virtual void set_time(int index, double new_time) {
    assert(index == 0);
    time_ = new_time;
  }

  virtual int num_independent_flavors() const { return 2 * Rank; }

  virtual int get_flavor(int index) const {
    assert(index >= 0 && index < 2 * Rank);
    return flavors_[index];
  }

  virtual void set_flavor(int index, int new_flavor) {
    assert(index >= 0 && index < 2 * Rank);
    flavors_[index] = new_flavor;
  }

  virtual const std::vector<int> &get_time_index(int flavor_index) const {
    assert(flavor_index >= 0 && flavor_index < 2 * Rank);
    return time_index_;
  }

  virtual bool operator==(const EqualTimeGWorm<Rank> &other_worm) const {
    return (time_ == other_worm.time_ && flavors_ == other_worm.flavors_);
  }

  const std::string& get_name() const {
    return name_;
  }

 private:
  std::string name_;
  double time_;
  boost::array<int, 2 * Rank> flavors_;
  std::vector<int> time_index_;
};

#include "worm.ipp"
