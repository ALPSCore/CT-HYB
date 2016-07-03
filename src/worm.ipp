#include "../worm.hpp"

template<unsigned int NumTimes>
std::vector <psi> CorrelationWorm<NumTimes>::get_operators() const {
  std::vector <psi> ops
  for (int it = 0; it < NumTimes; ++it) {
    //creation operator
    ops.push_back(
        psi(
            OperatorTime(times_[it], 200), CREATION_OP, flavors_[it][0]
        )
    );
    //annihilation operator
    ops.push_back(
        psi(
            OperatorTime(times_[it], 100), ANNIHILATION_OP, flavors_[it][1]
        )
    );
  }
  return ops;
}

