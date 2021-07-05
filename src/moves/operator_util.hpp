#pragma once

#include <iostream>
#include <utility>

#include <boost/assert.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/array.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/if.hpp>

#include "../model/operator.hpp"

/**
* Check consistency of operators, creation_operators, annihilation_operators
*
* @param operators a list of creation and annihilation operators
* @param creation_operators a list of creation operators
* @param annihilation_operators a list of annihilation operators
*/
void check_consistency_operators(const operator_container_t &operators,
                                 const operator_container_t &creation_operators,
                                 const operator_container_t &annihilation_operators);


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
int count_num_pairs(const operator_container_t &c_operators,
                    const operator_container_t &a_operators,
                    int flavor_ins,
                    int flavor_rem,
                    double t1,
                    double t2,
                    double distance);

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
//boost::tuple<int,operator_container_t::iterator,operator_container_t::iterator>
//pick_up_pair(alps::random01& rng, const operator_container_t& c_operators, const operator_container_t& a_operators, int flavor_ins, int flavor_rem, double t1, double t2, double distance, double BETA);
template<typename R>
boost::tuple<int, operator_container_t::iterator, operator_container_t::iterator>
pick_up_pair(R &rng,
             const operator_container_t &c_operators,
             const operator_container_t &a_operators,
             int flavor_ins,
             int flavor_rem,
             double t1,
             double t2,
             double distance,
             double BETA) {
  namespace bll = boost::lambda;

  typedef operator_container_t::iterator it_t;

  //get views to operators in the window
  double tau_low = std::min(t1, t2);
  double tau_high = std::max(t1, t2);
  std::pair<operator_container_t::iterator, operator_container_t::iterator>
      crange = c_operators.range(tau_low <= bll::_1, bll::_1 <= tau_high);
  std::pair<operator_container_t::iterator, operator_container_t::iterator>
      arange = a_operators.range(tau_low <= bll::_1, bll::_1 <= tau_high);

  typedef std::list<std::pair<it_t, it_t> > pairs_t;
  pairs_t pairs;
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
        pairs.push_back(std::make_pair(it_c, it_a));
        ++num_pairs;
      }
    }
  }

  if (num_pairs > 0) {
    int pos = (int) (rng() * num_pairs);
    pairs_t::iterator it_p = pairs.begin();
    std::advance(it_p, pos);
    return boost::make_tuple(num_pairs, (*it_p).first, (*it_p).second);
  } else {
    return boost::make_tuple(0, c_operators.end(), a_operators.end());
  }
}

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
                                 bool &error);

/**
 * @brief Count the number of operators in a given time window (min[t1,t2], max[t1,t2])
 * @param operators set of operators
 * @param t1 an end point of the window
 * @param t2 an end point of the window
 */
template<typename T>
int num_operators_in_range_open(const operator_container_t &operators, T t1, T t2) {
  namespace bll = boost::lambda;
  typedef operator_container_t::iterator Iterator;

  std::pair<Iterator, Iterator> ops_range
      = operators.range(
          std::min(t1, t2) < bll::_1, bll::_1 < std::max(t1, t2)
      );
  return std::distance(ops_range.first, ops_range.second);
}

/**
 * Spread of a set of operators in the imaginary time (without taking into accout beta-periodicity)
 * The spread is as the distance betwen the operator with the largest tau and the one with the smallest tau.
*/
double compute_spread_non_periodic(const std::vector<psi> &ops);


/**
 * Count the number of operators for each falvor
 */
template<typename InputItr>
void count_operators(InputItr begin,
                     InputItr end,
                     int num_flavors,
                     std::vector<int> &num_ops_flavors);

/**
 * Count the number of operators of a given type for each falvor
 */
template<typename InputItr>
void count_operators(InputItr begin,
                     InputItr end,
                     int num_flavors,
                     std::vector<int> &num_ops_flavors,
                     OPERATOR_TYPE type);

/**
 * Count the number of operators in the interval [tau_low, tau_high]
 */
//template<typename InputContainer>
//void count_operators_closed(const InputContainer &ops,
                            //double tau_low,
                            //double tau_high,
                            //int num_flavors,
                            //std::vector<int> &num_ops_flavors);

/**
 * Count some operators from the container defined by begin and end.
 * The numbers of operators to be picked up are given by num_ops_flavors.
 * The operators selected are stored in ops_picked_up.
 * random01() returns a random number in [0,1).
 */
template<typename InputItr>
void pick_up_operators(InputItr begin,
                       InputItr end,
                       std::vector<int> &num_ops_flavors,
                       std::vector<psi>& ops_picked_up,
                       alps::random01& random01
);

#include "operator_util.ipp"