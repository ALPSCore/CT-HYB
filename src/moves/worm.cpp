#include "worm.hpp"
#include "worm.ipp"


template std::vector<psi> GWorm<1>::get_operators() const;
template std::vector<psi> GWorm<2>::get_operators() const;
template std::vector<psi> EqualTimeGWorm<1>::get_operators() const;
template std::vector<psi> EqualTimeGWorm<2>::get_operators() const;

template<>
std::vector<psi>
TwoPointCorrWorm<PH_CHANNEL>::get_operators() const {
  return {
    psi(OperatorTime(get_time(0), 1), CREATION_OP,     get_flavor(0)),
    psi(OperatorTime(get_time(0), 0), ANNIHILATION_OP, get_flavor(1)),
    psi(OperatorTime(get_time(1), 1), CREATION_OP,     get_flavor(2)),
    psi(OperatorTime(get_time(1), 0), ANNIHILATION_OP, get_flavor(3))
  };
}

template<>
std::vector<psi>
TwoPointCorrWorm<PP_CHANNEL>::get_operators() const {
  return {
    psi(OperatorTime(get_time(0), 1), ANNIHILATION_OP, get_flavor(0)),
    psi(OperatorTime(get_time(0), 0), ANNIHILATION_OP, get_flavor(1)),
    psi(OperatorTime(get_time(1), 1), CREATION_OP,     get_flavor(2)),
    psi(OperatorTime(get_time(1), 0), CREATION_OP,     get_flavor(3))
  };
}

template<>
ConfigSpaceEnum::Type
TwoPointCorrWorm<PH_CHANNEL>::get_config_space() const {
  return ConfigSpaceEnum::Two_point_PH;
}

template<>
ConfigSpaceEnum::Type
TwoPointCorrWorm<PP_CHANNEL>::get_config_space() const {
  return ConfigSpaceEnum::Two_point_PP;
}


template<>
std::vector<psi>
ThreePointCorrWorm<PH_CHANNEL>::get_operators() const {
  return {
    psi(OperatorTime(get_time(0), 1), ANNIHILATION_OP, get_flavor(0)),
    psi(OperatorTime(get_time(1), 0), CREATION_OP,     get_flavor(1)),
    psi(OperatorTime(get_time(2), 1), CREATION_OP,     get_flavor(2)),
    psi(OperatorTime(get_time(2), 0), ANNIHILATION_OP, get_flavor(3))
  };
}

template<>
std::vector<psi>
ThreePointCorrWorm<PP_CHANNEL>::get_operators() const {
  return {
    psi(OperatorTime(get_time(0), 1), ANNIHILATION_OP, get_flavor(0)),
    psi(OperatorTime(get_time(1), 0), ANNIHILATION_OP, get_flavor(1)),
    psi(OperatorTime(get_time(2), 1), CREATION_OP,     get_flavor(2)),
    psi(OperatorTime(get_time(2), 0), CREATION_OP,     get_flavor(3))
  };
}

template<>
ConfigSpaceEnum::Type
ThreePointCorrWorm<PH_CHANNEL>::get_config_space() const {
  return ConfigSpaceEnum::Three_point_PH;
}

template<>
ConfigSpaceEnum::Type
ThreePointCorrWorm<PP_CHANNEL>::get_config_space() const {
  return ConfigSpaceEnum::Three_point_PP;
}