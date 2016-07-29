#include "operator_util.hpp"

template<typename InputItr>
void count_operators(InputItr begin,
                     InputItr end,
                     int num_flavors,
                     std::vector<int> &num_ops_flavors) {
  num_ops_flavors.resize(num_flavors);
  std::fill(num_ops_flavors.begin(), num_ops_flavors.end(), 0);
  for (InputItr it = begin; it != end; ++it) {
    ++ num_ops_flavors[it->flavor()];
  }
}

template<typename InputItr>
void count_operators(InputItr begin,
                     InputItr end,
                     int num_flavors,
                     std::vector<int> &num_ops_flavors,
                     OPERATOR_TYPE type) {
  num_ops_flavors.resize(num_flavors);
  std::fill(num_ops_flavors.begin(), num_ops_flavors.end(), 0);
  for (InputItr it = begin; it != end; ++it) {
    if (it->type() == type) {
      ++ num_ops_flavors[it->flavor()];
    }
  }
}

//void count_operators_closed(const operator_container_t &ops,
                            //double tau_low,
                            //double tau_high,
                            //int num_flavors,
                            //std::vector<int> &num_ops_flavors) {
  //namespace bll = boost::lambda;
//
  //num_ops_flavors.resize(num_flavors);
  //std::fill(num_ops_flavors.begin(), num_ops_flavors.end(), 0);
//
  //std::pair<operator_container_t::iterator,operator_container_t::iterator> ops_range = ops.range(
      //tau_low <= bll::_1, bll::_1 <= tau_high
  //);
  //for (operator_container_t::const_iterator it = ops_range.first; it != ops_range.second; ++it) {
    //++ num_ops_flavors[it->flavor()];
  //}
//}

template<typename InputItr>
void pick_up_operators(InputItr begin,
                       InputItr end,
                       std::vector<int> &num_ops_flavors,
                       std::vector<psi>& ops_picked_up,
                       alps::random01& random01
) {
  const int num_flavors = num_ops_flavors.size();

  std::vector<psi> ops(begin, end), ops_flavor;

  for (int flavor = 0; flavor < num_flavors; ++ flavor) {
    if (num_ops_flavors[flavor] == 0) {
      continue;
    }
    ops_flavor.resize(0);
    for (std::vector<psi>::iterator it = ops.begin(); it != ops.end(); ++it) {
      if (it->flavor() == flavor) {
        ops_flavor.push_back(*it);
      }
    }
    if (num_ops_flavors[flavor] > ops_flavor.size()) {
      throw std::runtime_error("num_ops_flavors[flavor] > ops_flavor.size()");
    }

    //shuffle elements using the same alogrithm used in std::shuffle
    for (int iop = 0; iop < num_ops_flavors[flavor]; ++iop) {
      std::swap(
          ops_flavor[iop],
          ops_flavor[iop + static_cast<int>(random01() * (ops_flavor.size() - iop))]
      );
    }

    //copy the first num_ops_flavors[flavor] elements
    std::copy(ops_flavor.begin(), ops_flavor.begin() + num_ops_flavors[flavor], std::back_inserter(ops_picked_up));
  }
}
