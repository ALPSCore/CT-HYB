#include "worm.hpp"
#include "worm.ipp"


template std::vector<psi> GWorm<1,false>::get_operators() const;
template std::vector<psi> GWorm<2,false>::get_operators() const;
template std::vector<psi> GWorm<1,true>::get_operators() const;
template std::vector<psi> GWorm<2,true>::get_operators() const;
template std::vector<psi> EqualTimeGWorm<1>::get_operators() const;
template std::vector<psi> EqualTimeGWorm<2>::get_operators() const;

template class ThreePointCorrWorm<PH_CHANNEL,true>;
template class ThreePointCorrWorm<PP_CHANNEL,true>;
template class ThreePointCorrWorm<PH_CHANNEL,false>;
template class ThreePointCorrWorm<PP_CHANNEL,false>;

/*
 * Gworm (G1, G2, vartheta, h)
 */
template<>
ConfigSpaceEnum::Type
GWorm<1,false>::get_config_space() const {
  return ConfigSpaceEnum::G1;
}

template<>
ConfigSpaceEnum::Type
GWorm<2,false>::get_config_space() const {
  return ConfigSpaceEnum::G2;
}

template<>
ConfigSpaceEnum::Type
GWorm<1,true>::get_config_space() const {
  return ConfigSpaceEnum::vartheta;
}

template<>
ConfigSpaceEnum::Type
GWorm<2,true>::get_config_space() const {
  return ConfigSpaceEnum::h;
}

/*
 * TwoPointCorrWorm
 */
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
  return ConfigSpaceEnum::lambda;
}

template<>
ConfigSpaceEnum::Type
TwoPointCorrWorm<PP_CHANNEL>::get_config_space() const {
  return ConfigSpaceEnum::varphi;
}


/*
 * ThreePointCorrWorm
 */
template<>
ConfigSpaceEnum::Type
ThreePointCorrWorm<PH_CHANNEL,false>::get_config_space() const {
  return ConfigSpaceEnum::Three_point_PH;
}

template<>
ConfigSpaceEnum::Type
ThreePointCorrWorm<PP_CHANNEL,false>::get_config_space() const {
  return ConfigSpaceEnum::Three_point_PP;
}


template<>
ConfigSpaceEnum::Type
ThreePointCorrWorm<PH_CHANNEL,true>::get_config_space() const {
  return ConfigSpaceEnum::eta;
}

template<>
ConfigSpaceEnum::Type
ThreePointCorrWorm<PP_CHANNEL,true>::get_config_space() const {
  return ConfigSpaceEnum::gamma;
}


template<>
std::vector<psi>
ThreePointCorrWorm<PH_CHANNEL,false>::get_operators() const {
  return {
    psi(OperatorTime(get_time(0), 1), ANNIHILATION_OP, get_flavor(0), false),
    psi(OperatorTime(get_time(1), 0), CREATION_OP,     get_flavor(1), false),
    psi(OperatorTime(get_time(2), 1), CREATION_OP,     get_flavor(2)),
    psi(OperatorTime(get_time(2), 0), ANNIHILATION_OP, get_flavor(3))
  };
}

template<>
std::vector<psi>
ThreePointCorrWorm<PP_CHANNEL,false>::get_operators() const {
  return {
    psi(OperatorTime(get_time(0), 1), ANNIHILATION_OP, get_flavor(0), false),
    psi(OperatorTime(get_time(1), 0), ANNIHILATION_OP, get_flavor(1), false),
    psi(OperatorTime(get_time(2), 1), CREATION_OP,     get_flavor(2)),
    psi(OperatorTime(get_time(2), 0), CREATION_OP,     get_flavor(3))
  };
}

template<>
std::vector<psi>
ThreePointCorrWorm<PH_CHANNEL,true>::get_operators() const {
  return {
    psi(OperatorTime(get_time(0), 1), ANNIHILATION_OP, get_flavor(0), true),
    psi(OperatorTime(get_time(1), 0), CREATION_OP,     get_flavor(1), true),
    psi(OperatorTime(get_time(2), 1), CREATION_OP,     get_flavor(2)),
    psi(OperatorTime(get_time(2), 0), ANNIHILATION_OP, get_flavor(3))
  };
}

template<>
std::vector<psi>
ThreePointCorrWorm<PP_CHANNEL,true>::get_operators() const {
  return {
    psi(OperatorTime(get_time(0), 1), ANNIHILATION_OP, get_flavor(0), true),
    psi(OperatorTime(get_time(1), 0), ANNIHILATION_OP, get_flavor(1), true),
    psi(OperatorTime(get_time(2), 1), CREATION_OP,     get_flavor(2)),
    psi(OperatorTime(get_time(2), 0), CREATION_OP,     get_flavor(3))
  };
}