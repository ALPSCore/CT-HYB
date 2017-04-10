#include "gtest.h"

#include <ir_basis/ir_basis.hpp>

#include <boost/math/special_functions/bessel.hpp>
#include <boost/multi_array.hpp>

//#include <boost/timer/timer.hpp>
//#include <time.h>


void compute_Tnl_legendre(int n_matsubara, int n_legendre, boost::multi_array<std::complex<double>,2> &Tnl) {
  double sign_tmp = 1.0;
  Tnl.resize(boost::extents[n_matsubara][n_legendre]);
  for (int im = 0; im < n_matsubara; ++im) {
    std::complex<double> ztmp(0.0, 1.0);
    for (int il = 0; il < n_legendre; ++il) {
      Tnl[im][il] = sign_tmp * ztmp * std::sqrt(2 * il + 1.0) * boost::math::sph_bessel(il, 0.5 * (2 * im + 1) * M_PI);
      ztmp *= std::complex<double>(0.0, 1.0);
    }
    sign_tmp *= -1;
  }
}

