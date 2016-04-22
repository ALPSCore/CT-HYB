#pragma once

#include <alps/mc/random01.hpp>

#include <boost/assert.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/array.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/if.hpp>

enum OPERATOR_TYPE {
  CREATION_OP = 0,
  ANNIHILATION_OP = 1,
  INVALID_OP = 2,
};


//an operator.
//operators are described by the time where they are inserted, as well as their site, flavor, and type (creation/annihilation).
class psi 
{
 public:
  psi() {t_=0; type_=INVALID_OP; flavor_=0; site_=0;};
  psi(double t, OPERATOR_TYPE type, int flavor) {t_=t; type_=type; flavor_=flavor; site_=0;};
  double time() const {return t_;}

  int flavor() const {return flavor_;}  
  OPERATOR_TYPE type() const {return type_;} // 0=create, 1=destroy
  void set_time(double t) {t_ = t;}

  void set_flavor(int flavor) {flavor_ = flavor;}
  void set_type(OPERATOR_TYPE type) {type_ = type;}
 private:
  double t_;
  int site_, flavor_;
  OPERATOR_TYPE type_;
};

inline bool operator<(const psi& t1, const psi& t2) {
  return t1.time() < t2.time();
}

inline bool operator<(const psi& t1, const double t2) {
  return t1.time() < t2;
}

inline bool operator<(const double t1, const psi& t2) {
  return t1 < t2.time();
}

inline bool operator<=(const psi& t1, const double t2) {
  return t1.time() <= t2;
}

inline bool operator<=(const double t1, const psi& t2) {
  return t1 <= t2.time();
}

inline bool operator>(const psi& t1, const psi& t2) {
 return  t1.time() > t2.time();
}

inline bool operator==(const psi& op1, const psi& op2) {
  return op1.time()==op2.time() && op1.type() == op2.type() && op1.flavor()==op2.flavor();
}

inline bool operator!=(const psi& op1, const psi& op2) {
  return !(op1==op2);
}

typedef boost::multi_index::multi_index_container<psi> operator_container_t; //one can use range() with multi_index_container.

inline void print_list(const operator_container_t &operators)
{
  std::cout<<"list: "<<std::endl;
  for(operator_container_t::const_iterator it=operators.begin(); it!=operators.end(); ++it) {
    std::cout<<it->time()<<" ";
  }
  std::cout<<std::endl;
}

inline void safe_erase(operator_container_t& operators, const psi& op) {
  operator_container_t::iterator it_target = operators.find(op);
  if (it_target==operators.end()) {
    throw std::runtime_error("Error in safe_erase: op is not found.");
  }
  operators.erase(it_target);
}

inline std::pair<operator_container_t::iterator, bool> safe_insert(operator_container_t& operators, const psi& op) {
  //const bool err = operators.find(op)!=operators.end() ? true : false;

  std::pair<operator_container_t::iterator, bool> r = operators.insert(op);
  if(!r.second) {
    print_list(operators);
    std::cerr << "Trying to insert an operator at " << op.time() << " " << op.type() << " " << op.flavor() << std::endl;
    //std::cerr << "err " << err << std::endl;
    throw std::runtime_error("problem, cannot insert a operator");
  }
  return r;
}

/**
* Check consistency of operators, creation_operators, annihilation_operators
*
* @param operators a list of creation and annihilation operators
* @param creation_operators a list of creation operators
* @param annihilation_operators a list of annihilation operators
*/
void check_consistency_operators(const operator_container_t& operators,
  const operator_container_t& creation_operators,
  const operator_container_t& annihilation_operators);


/**
* Count the number of pairs of creation and annihilation operators in a given time window
*
* @param c_operators the list of creation operators
* @param a_operators the list of annihilation operators
* @param flavor_ins the flavor of creation operators of the pairs
* @param flavor_rem the flavor of annihilation operators of the pairs
* @param t1 upper bound or lower bound of the time window
* @param t2 upper bound or lower bound of the time window
* @param distance cutoff for the mutual distance of a pair of creation and annihilation operators
*/
int count_num_pairs(const operator_container_t& c_operators, const operator_container_t& a_operators, int flavor_ins, int flavor_rem, double t1, double t2, double distance);

/**
* Pick up a pair of creation and annihilation operators in a given time window and returns interators pointing to the picked-up operators
*
* @param pseudo-random-number generator
* @param c_operators the list of creation operators
* @param a_operators the list of annihilation operators
* @param flavor_ins the flavor of creation operators of the pairs
* @param flavor_rem the flavor of annihilation operators of the pairs
* @param t1 upper bound or lower bound of the time window
* @param t2 upper bound or lower bound of the time window
* @param distance cutoff for the mutual distance of a pair of creation and annihilation operators
* @param BETA inverse temperature
*/
boost::tuple<int,operator_container_t::iterator,operator_container_t::iterator>
pick_up_pair(alps::random01& rng, const operator_container_t& c_operators, const operator_container_t& a_operators, int flavor_ins, int flavor_rem, double t1, double t2, double distance, double BETA);

/**
* Count the number of pairs of creation and annihilation operators in a given time window after the insertion of a creation operator and an annihilation operator
*
* @param c_operators the list of creation operators
* @param a_operators the list of annihilation operators
* @param flavor_ins the flavor of creation operators of the pairs
* @param flavor_rem the flavor of annihilation operators of the pairs
* @param t1 upper bound or lower bound of the time window
* @param t2 upper bound or lower bound of the time window
* @param distance cutoff for the mutual distance of a pair of creation and annihilation operators
* @param tau_ins the imaginary time where the new creation operator is inserted
* @param tau_rem the imaginary time where the new annihilation operator is inserted
* @param error if and only if there are already some operators at tau_in or tau_rem, error is set to true.
*/
int count_num_pairs_after_insert(const operator_container_t& operators, const operator_container_t& creation_operators, const operator_container_t& annihilation_operators,
                                 int flavor_ins, int flavor_rem,
                                 double t1, double t2, double distance, double tau_ins, double tau_rem, bool& error);

//c^¥dagger(flavor0) c(flavor1) c^¥dagger(flavor2) c(flavor3) ... at the equal time
template<int N>
class EqualTimeOperator {
public:
  EqualTimeOperator() : time_(-1.0) {
    std::fill(flavors_.begin(),flavors_.end(),-1);
  };

  EqualTimeOperator(const boost::array<int,2*N>& flavors, double time=-1.0) : flavors_(flavors), time_(time) {};

  EqualTimeOperator(const int* flavors, double time=-1.0) : time_(time) {
    for (int i=0; i<N; ++i) {
      flavors_[i] = flavors[i];
    }
  };

  inline int flavor(int idx) const {
    assert(idx>=0 && idx<2*N);
    return flavors_[idx];
  }

  inline double get_time() const {return time_;}

private:
  boost::array<int,2*N> flavors_;
  double time_;
};

typedef EqualTimeOperator<1> CdagC;

template<int N>
inline bool operator<(const EqualTimeOperator<N>& op1, const EqualTimeOperator<N>& op2) {
  for (int idigit=0; idigit<N; ++idigit) {
    if (op1.flavor(idigit)<op2.flavor(idigit)) {
      return true;
    } else if (op1.flavor(idigit)>op2.flavor(idigit)) {
      return false;
    }
  }
  return false;
}
