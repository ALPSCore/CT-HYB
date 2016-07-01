#pragma once

#include "operator.hpp"

class Worm {
 public:

  /** Get creation and annihilation operators (not time-ordered)*/
  const std::vector<psi>& get_operators() const = 0;
};

template<unsigned int NumTimes>
class CorretionWorm {
 public:
  CorretionWorm(const std::vector<int>& flavors, const std::vector<double>& times);

  const std::vector<psi>& get_operators() const;

 private:
};
