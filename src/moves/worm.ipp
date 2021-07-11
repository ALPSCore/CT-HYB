#include "worm.hpp"


/**
 * Returns c c^dagger ,..., c c^dagger
 */
template<unsigned int Rank>
std::vector<psi> GWorm<Rank>::get_operators() const {
  std::vector<psi> ops;
  for (int it = 0; it < Rank; ++it) {
    //annihilation operator
    ops.push_back(
        psi(
            OperatorTime(times_[2 * it], 0), ANNIHILATION_OP, flavors_[2 * it]
        )
    );
    //creation operator
    ops.push_back(
        psi(
            OperatorTime(times_[2 * it + 1], 0), CREATION_OP, flavors_[2 * it + 1]
        )
    );
  }
  return ops;
}