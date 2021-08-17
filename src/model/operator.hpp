#pragma once

#include <iostream>
#include <cassert>
#include <vector>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/array.hpp>

#include "../common/logger.hpp"

enum OPERATOR_TYPE {
  CREATION_OP = 0,
  ANNIHILATION_OP = 1,
  INVALID_OP = 2,
};

enum ActionType {
  INSERTION,
  REMOVAL,
};

//Class present the imaginary time of an oprator
template<class T>
class OperatorTimeTemplate {
 public:
  OperatorTimeTemplate() : time_(0), small_idx_(0) { }
  OperatorTimeTemplate(T time) : time_(time), small_idx_(0) { }
  OperatorTimeTemplate(T time, int small_idx) : time_(time), small_idx_(small_idx) { }

  inline T time() const { return time_; }
  inline int small_index() const { return small_idx_; }

  inline void set_time(T time) { time_ = time; }
  inline void set_small_index(int small_idx) { small_idx_ = small_idx; }

 private:
  T time_;
  int small_idx_;
};

typedef OperatorTimeTemplate<double> OperatorTime;

template<class T>
std::ostream &operator<<(std::ostream &os, const OperatorTimeTemplate<T> &t1) {
  os << "( " << t1.time() << " , " << t1.small_index() << " )";
  return os;
}

template<class T>
bool operator<(const OperatorTimeTemplate<T> &t1, const OperatorTimeTemplate<T> &t2) {
  if (t1.time() == t2.time()) {
    return (t1.small_index() < t2.small_index());
  } else {
    return (t1.time() < t2.time());
  }
}

template<class T>
bool operator<=(const OperatorTimeTemplate<T> &t1, const OperatorTimeTemplate<T> &t2) {
  if (t1.time() == t2.time()) {
    return (t1.small_index() <= t2.small_index());
  } else {
    return (t1.time() <= t2.time());
  }
}

template<class T>
bool operator>(const OperatorTimeTemplate<T> &t1, const OperatorTimeTemplate<T> &t2) {
  return t2 < t1;
}

template<class T>
bool operator>=(const OperatorTimeTemplate<T> &t1, const OperatorTimeTemplate<T> &t2) {
  return t2 <= t1;
}

template<class T>
bool operator==(const OperatorTimeTemplate<T> &t1, const OperatorTimeTemplate<T> &t2) {
  return (t1.time() == t2.time()) && (t1.small_index() == t2.small_index());
}

template<class T>
double operator-(const OperatorTimeTemplate<T> &t1, const OperatorTimeTemplate<T> &t2) {
  return t1.time() - t2.time();
}

template<class T>
double operator-(const OperatorTimeTemplate<T> &t1, const double &t2) {
  return t1.time() - t2;
}

template<class T>
double operator-(const double &t1, const OperatorTimeTemplate<T> &t2) {
  return t1 - t2.time();
}

template<class T>
double operator+(const OperatorTimeTemplate<T> &t1, const OperatorTimeTemplate<T> &t2) {
  return t1.time() + t2.time();
}

template<class T>
double operator+(const OperatorTimeTemplate<T> &t1, const double &t2) {
  return t1.time() + t2;
}

//an operator.
//operators are described by the time where they are inserted, as well as their site, flavor, and type (creation/annihilation).
class psi {
 public:
  typedef OperatorTime itime_type;
  typedef OperatorTime TIME_T;

  psi() : t_(0), type_(INVALID_OP), flavor_(0), time_deriv_(false) {
  };

  psi(TIME_T t, OPERATOR_TYPE type, int flavor, bool time_deriv=false)
    : t_(t), type_(type), flavor_(flavor), time_deriv_(time_deriv) {
  };

  TIME_T time() const { return t_; }
  int flavor() const { return flavor_; }
  bool time_deriv() const {return time_deriv_; }
  OPERATOR_TYPE type() const { return type_; } // 0=creation, 1=annihilation
  void set_time(TIME_T t) { t_ = t; }

  void set_flavor(int flavor) { flavor_ = flavor; }
  void set_type(OPERATOR_TYPE type) { type_ = type; }
  void set_time_deriv(bool time_deriv) { time_deriv_ = time_deriv; }

 private:
  TIME_T t_;
  int flavor_;
  OPERATOR_TYPE type_;
  bool time_deriv_;
};

//class LocalPhi : psi {
  //using psi::psi;
//};

inline OperatorTime operator_time(const psi &op) {
  return op.time();
}

inline int operator_flavor(const psi &op) {
  return op.flavor();
}

struct OperatorEqualTime {
  bool operator()(const psi &op1, const psi &op2) const {
    return op1.time() == op2.time();
  }
};

inline bool operator<(const psi &op1, const psi &op2) {
  return op1.time() < op2.time();
}

inline bool operator<(const psi &op1, const double t2) {
  return op1.time() < OperatorTime(t2);
}

inline bool operator<(const psi &op1, const OperatorTime t2) {
  return op1.time() < t2;
}

inline bool operator<(const double t1, const psi &t2) {
  return OperatorTime(t1) < t2.time();
}

inline bool operator<(const OperatorTime &t1, const double &t2) {
  return t1 < OperatorTime(t2, 0);
}

inline bool operator<(const OperatorTime &t1, const psi &t2) {
  return t1 < t2.time();
}

inline bool operator<=(const psi &t1, const double t2) {
  return t1.time() <= OperatorTime(t2);
}

inline bool operator<=(const psi &t1, const OperatorTime t2) {
  return t1.time() <= t2;
}

inline bool operator<=(const double t1, const psi &t2) {
  return OperatorTime(t1) <= t2.time();
}

inline bool operator<=(const OperatorTime t1, const psi &t2) {
  return t1 <= t2.time();
}

inline bool operator<=(const OperatorTime &t1, const double &t2) {
  return t1 <= OperatorTime(t2, 0);
}

inline bool operator<=(const double &t1, const OperatorTime &t2) {
  return OperatorTime(t1, 0) <= t2;
}

inline bool operator>(const psi &t1, const psi &t2) {
  return t1.time() > t2.time();
}

inline bool operator==(const psi &op1, const psi &op2) {
  return op1.time() == op2.time() && op1.type() == op2.type() && op1.flavor() == op2.flavor();
}

inline bool operator!=(const psi &op1, const psi &op2) {
  return !(op1 == op2);
}

typedef boost::multi_index::multi_index_container<psi>
    operator_container_t; //one can use range() with multi_index_container.

template<typename V>
void print_list(const V &operators) {
  logger_out << "list: " << std::endl;
  for (typename V::const_iterator it = operators.begin(); it != operators.end(); ++it) {
    logger_out << "time " << it->time() << "[" << it->flavor() << "]" << " ";
  }
  logger_out << std::endl;
}

inline void safe_erase(operator_container_t &operators, const psi &op) {
  operator_container_t::iterator it_target = operators.find(op);
  if (it_target == operators.end()) {
    throw std::runtime_error("Error in safe_erase: op is not found.");
  }
  operators.erase(it_target);
}

template<typename Iterator>
inline void safe_erase(operator_container_t &operators, Iterator first, Iterator last) {
  for (Iterator it = first; it != last; ++it) {
    safe_erase(operators, *it);
  }
}


inline void safe_erase(operator_container_t &operators, const std::vector<psi> &ops) {
  safe_erase(operators, ops.begin(), ops.end());
}

inline std::pair<operator_container_t::iterator, bool> safe_insert(operator_container_t &operators, const psi &op) {
  std::pair<operator_container_t::iterator, bool> r = operators.insert(op);
  if (!r.second) {
    print_list(operators);
    std::cerr << "Trying to insert an operator at " << op.time() << " " << op.type() << " " << op.flavor() << std::endl;
    throw std::runtime_error("problem, cannot insert a operator");
  }
  return r;
}

template<typename Iterator>
inline void safe_insert(operator_container_t &operators, Iterator first, Iterator last) {
  for (Iterator it = first; it != last; ++it) {
    safe_insert(operators, *it);
  }
}


inline void safe_insert(operator_container_t &operators, const std::vector<psi> &ops) {
  safe_insert(operators, ops.begin(), ops.end());
}



//c^¥dagger(flavor0) c(flavor1) c^¥dagger(flavor2) c(flavor3) ... at the equal time
template<int N>
class EqualTimeOperator {
 public:
  EqualTimeOperator() : time_(-1.0) {
    std::fill(flavors_.begin(), flavors_.end(), -1);
  };

  EqualTimeOperator(const boost::array<int, 2 * N> &flavors, double time = -1.0) : flavors_(flavors), time_(time) { };

  EqualTimeOperator(const int *flavors, double time = -1.0) : time_(time) {
    for (int i = 0; i < 2 * N; ++i) {
      flavors_[i] = flavors[i];
    }
  };

  inline int flavor(int idx) const {
    assert(idx >= 0 && idx < 2 * N);
    return flavors_[idx];
  }

  inline double get_time() const { return time_; }

 private:
  boost::array<int, 2 * N> flavors_;
  double time_;
};

typedef EqualTimeOperator<1> CdagC;

template<int N>
inline bool operator<(const EqualTimeOperator<N> &op1, const EqualTimeOperator<N> &op2) {
  for (int idigit = 0; idigit < N; ++idigit) {
    if (op1.flavor(idigit) < op2.flavor(idigit)) {
      return true;
    } else if (op1.flavor(idigit) > op2.flavor(idigit)) {
      return false;
    }
  }
  return false;
}


std::ostream &operator<<(std::ostream &os, const psi &psi);

std::ostream &operator<<(std::ostream &os, const operator_container_t &operators);

std::ostream &operator<<(std::ostream &os, const std::vector<psi> &operators);
