#include "worm.hpp"


/**
 * Returns c c^dagger ,..., c c^dagger
 */
template<unsigned int Rank, bool TIME_DERIV>
std::vector<psi> GWorm<Rank,TIME_DERIV>::get_operators() const {
  std::vector<psi> ops;
  for (int it = 0; it < Rank; ++it) {
    //annihilation operator
    ops.push_back(
      psi(
        OperatorTime(
          times_[2 * it], small_indices_[2 * it]),
          ANNIHILATION_OP, flavors_[2 * it], TIME_DERIV
      )
    );
    //creation operator
    ops.push_back(
      psi(
        OperatorTime(times_[2 * it + 1], small_indices_[2 * it + 1]),
        CREATION_OP, flavors_[2 * it + 1], TIME_DERIV
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