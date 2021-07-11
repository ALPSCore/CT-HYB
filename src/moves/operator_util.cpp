#include <alps/mc/random01.hpp>
#include "operator_util.hpp"

void check_consistency_operators(const operator_container_t &operators,
                                 const operator_container_t &creation_operators,
                                 const operator_container_t &annihilation_operators) {
#ifndef NDEBUG

  if (creation_operators.size() != annihilation_operators.size() || operators.size() != 2 * creation_operators.size()) {
    std::runtime_error("Error in check_consistency_operators: size inconsistency");
  }

  operator_container_t::iterator it_c = creation_operators.begin();
  operator_container_t::iterator it_a = annihilation_operators.begin();
  for (operator_container_t::iterator it = operators.begin(); it != operators.end(); ++it) {
    if (it->type() == CREATION_OP) {
      if (*it != *it_c) {
        std::runtime_error("Error in check_consistency_operators: inconsistency in creation_operators");
      }
      ++it_c;
    } else {
      if (*it != *it_a) {
        std::runtime_error("Error in check_consistency_operators: inconsistency in annihilation_operators");
      }
      ++it_a;
    }
  }
#endif
}


int count_num_pairs(const operator_container_t &c_operators,
                    const operator_container_t &a_operators,
                    int flavor_ins,
                    int flavor_rem,
                    double t1,
                    double t2,
                    double distance) {
  namespace bll = boost::lambda;

  //get views to operators in the window
  double tau_low = std::min(t1, t2);
  double tau_high = std::max(t1, t2);
  std::pair<operator_container_t::iterator, operator_container_t::iterator>
      crange = c_operators.range(tau_low <= bll::_1, bll::_1 <= tau_high);
  std::pair<operator_container_t::iterator, operator_container_t::iterator>
      arange = a_operators.range(tau_low <= bll::_1, bll::_1 <= tau_high);

  int num_pairs = 0;
  for (operator_container_t::iterator it_c = crange.first; it_c != crange.second; ++it_c) {
    if (it_c->flavor() != flavor_ins) {
      continue;
    }
    for (operator_container_t::iterator it_a = arange.first; it_a != arange.second; ++it_a) {
      if (it_a->flavor() != flavor_rem) {
        continue;
      }
      if (std::abs(it_c->time() - it_a->time()) <= distance) {
        ++num_pairs;
      }
    }
  }

  return num_pairs;
}


int count_num_pairs_after_insert(const operator_container_t &operators,
                                 const operator_container_t &creation_operators,
                                 const operator_container_t &annihilation_operators,
                                 int flavor_ins,
                                 int flavor_rem,
                                 double t1,
                                 double t2,
                                 double distance,
                                 double tau_ins,
                                 double tau_rem,
                                 bool &error) {
  namespace bll = boost::lambda;
  typedef operator_container_t::iterator it_t;
  typedef operator_container_t::value_type op_t;

  error = false;

  int num_pairs = count_num_pairs(creation_operators, annihilation_operators, flavor_ins, flavor_rem, t1, t2, distance);
  if (std::abs(tau_ins - tau_rem) <= distance) {
    ++num_pairs;
  }

  if (operators.find(op_t(tau_ins, CREATION_OP, flavor_ins)) != operators.end()) {
    error = true;
    std::cerr << "creation operator already exists at tau_ins = " << tau_ins << std::endl;
    print_list(operators);
    return 0;
  }
  if (operators.find(op_t(tau_rem, ANNIHILATION_OP, flavor_rem)) != operators.end()) {
    error = true;
    std::cerr << "annihilation operator already exists at tau_rem = " << tau_rem << std::endl;
    print_list(operators);
    return 0;
  }

  //get views to operators in the window
  double tau_low = std::min(t1, t2);
  double tau_high = std::max(t1, t2);
  std::pair<it_t, it_t> range = operators.range(tau_low <= bll::_1, bll::_1 <= tau_high);
  for (it_t it_c = range.first; it_c != range.second; ++it_c) {
    if (it_c->flavor() != flavor_ins || it_c->type() != CREATION_OP) {
      continue;
    }
    if (std::abs(it_c->time() - tau_rem) <= distance) {
      ++num_pairs;
    }
  }
  for (it_t it_a = range.first; it_a != range.second; ++it_a) {
    if (it_a->flavor() != flavor_rem || it_a->type() != ANNIHILATION_OP) {
      continue;
    }
    if (std::abs(it_a->time() - tau_ins) <= distance) {
      ++num_pairs;
    }
  }

  return num_pairs;
}

std::ostream &operator<<(std::ostream &os, const psi &psi) {
  os << " [flavor " << psi.flavor() << " , " << psi.time() << " ] ";
  return os;
}

std::ostream &operator<<(std::ostream &os, const operator_container_t &operators) {
  for (operator_container_t::iterator it = operators.begin(); it != operators.end(); ++it) {
    os << *it;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const std::vector<psi> &operators) {
  for (std::vector<psi>::const_iterator it = operators.begin(); it != operators.end(); ++it) {
    os << *it;
  }
  return os;
}

double compute_spread_non_periodic(const std::vector<psi> &ops) {
  if (ops.size()) {
    return 0.0;
  }
  double max_tau, min_tau;
  max_tau = min_tau = ops[0].time().time();
  for (std::vector<psi>::const_iterator it = ops.begin(); it != ops.end(); ++it) {
    max_tau = std::max(max_tau, it->time().time());
    min_tau = std::min(min_tau, it->time().time());
  }
  return max_tau - min_tau;
}


int count_ops_at(const operator_container_t &operators, double t) {
  int counter = 0;
  for (auto it=operators.begin(); it!=operators.end(); ++it) {
    if (it->time().time() == t) {
      counter += 1;
    }
  }
  return counter;
}