#include <alps/params.hpp>

#include <boost/random.hpp>

#include<Eigen/Dense>
#include<Eigen/LU>

//To avoid compiler errors for intel compiler on a cray machine
#define GTEST_USE_OWN_TR1_TUPLE 1

#include "gtest.h"

#include <alps/fastupdate/detail/util.hpp>
#include "../src/model/model.hpp"
#include "../src/util.hpp"

template<typename T>
boost::tuple<int,int,int,int,T>
get_tuple(int o0, int o1, int o2, int o3, int spin, int spin2, T val, int sites) {
  return boost::make_tuple(o0+spin*sites, o1+spin2*sites, o2+spin2*sites, o3+spin*sites, val);
};
