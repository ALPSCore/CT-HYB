#include "worm.hpp"

template<unsigned int NumTimes>
std::vector<psi> CorrelationWorm<NumTimes>::get_operators() const {
  std::vector<psi> ops;
  for (int it = 0; it < NumTimes; ++it) {
    //creation operator
    ops.push_back(
        psi(
            OperatorTime(times_[it], 1), CREATION_OP, flavors_[2 * it]
        )
    );
    //annihilation operator
    ops.push_back(
        psi(
            OperatorTime(times_[it], 0), ANNIHILATION_OP, flavors_[2 * it + 1]
        )
    );
  }
  return ops;
}

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

/**
 * Returns c^dagger c ,..., c^dagger c
 */
template<unsigned int Rank>
std::vector<psi> EqualTimeGWorm<Rank>::get_operators() const {
  std::vector<psi> ops;
  int small_idx = 2 * Rank - 1;
  for (int it = 0; it < Rank; ++it) {
    //creation operator
    ops.push_back(
        psi(
            OperatorTime(time_, small_idx), CREATION_OP, flavors_[2 * it]
        )
    );
    -- small_idx;

    //annihilation operator
    ops.push_back(
        psi(
            OperatorTime(time_, small_idx), ANNIHILATION_OP, flavors_[2 * it + 1]
        )
    );
    -- small_idx;

  }
  assert (small_idx == -1);
  return ops;
}
