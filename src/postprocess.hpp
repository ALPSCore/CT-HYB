#pragma once

#include <complex>

#include <Eigen/Dense>

#include <alps/gf/gf.hpp>
#include <alps/gf/tail.hpp>

#include "legendre.hpp"

template<typename T>
struct to_complex {
  std::complex<T> operator()(const T &re, const T &im) {
    return std::complex<T>(re, im);
  }
};

/**
 * Evaluate imaginary and Matsubara Green's functions from coefficients of Legendre polynominals
 *
 * @param result ALPS Monte Carlo result
 * @param parms ALPS parameters (input)
 * @param ar ALPS HDF5 archive (output)
 */
template<typename SOLVER_TYPE>
void compute_greens_functions(const typename alps::results_type<SOLVER_TYPE>::type &results,
                              const typename alps::parameters_type<SOLVER_TYPE>::type &parms, alps::hdf5::archive ar) {
  namespace g=alps::gf;

  const int n_tau(parms["N_TAU"]);
  const int n_site(parms["SITES"]);
  const int n_spin(parms["SPINS"]);
  const double beta(parms["BETA"]);
  const double temperature(1.0 / beta);
  const int n_matsubara(n_tau);
  const int n_legendre(parms["N_LEGENDRE_MEASUREMENT"].template as<int>());
  const int n_flavors = n_site * n_spin;

  const double sign = results["Sign"].template mean<double>();

  /*
   * Compute legendre coefficients in the original basis.
   */
  const std::vector<double> Gl_Re = results["Greens_legendre_Re"].template mean<std::vector<double> >();
  const std::vector<double> Gl_Im = results["Greens_legendre_Im"].template mean<std::vector<double> >();
  assert(Gl_Re.size() == n_flavors * n_flavors * n_legendre);
  boost::multi_array<std::complex<double>, 3> Gl(boost::extents[n_flavors][n_flavors][n_legendre]);
  std::transform(Gl_Re.begin(), Gl_Re.end(), Gl_Im.begin(), Gl.origin(), to_complex<double>());

  /*
   * Initialize LegendreTransformer
   */
  LegendreTransformer legendre_transformer(n_matsubara, n_legendre);

  /*
   * Compute G(tau) from Legendre coefficients
   */
  typedef alps::gf::three_index_gf<std::complex<double>, alps::gf::itime_mesh,
                                   alps::gf::index_mesh,
                                   alps::gf::index_mesh
  > ITIME_GF;

  ITIME_GF
      itime_gf(alps::gf::itime_mesh(beta, n_tau + 1), alps::gf::index_mesh(n_flavors), alps::gf::index_mesh(n_flavors));
  std::vector<double> Pvals(n_legendre);
  const std::vector<double> &sqrt_array = legendre_transformer.get_sqrt_2l_1();
  for (int itau = 0; itau < n_tau + 1; ++itau) {
    const double tau = itau * (beta / n_tau);
    const double x = 2 * tau / beta - 1.0;
    legendre_transformer.compute_legendre(x, Pvals); //Compute P_l[x]

    for (int flavor = 0; flavor < n_flavors; ++flavor) {
      for (int flavor2 = 0; flavor2 < n_flavors; ++flavor2) {
        for (int il = 0; il < n_legendre; ++il) {
          itime_gf(g::itime_index(itau), g::index(flavor), g::index(flavor2)) +=
              Pvals[il] * Gl[flavor][flavor2][il] * sqrt_array[il] * temperature / sign;
        }
      }
    }
  }
  itime_gf.save(ar, "/gtau");

  /*
   * Compute Gomega from Legendre coefficients
   */
  typedef alps::gf::three_index_gf<std::complex<double>, alps::gf::matsubara_positive_mesh,
                                   alps::gf::index_mesh,
                                   alps::gf::index_mesh
  > GOMEGA;

  const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &Tnl(legendre_transformer.Tnl());
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> tmp_mat(n_legendre, 1), tmp_mat2(n_tau, 1);
  GOMEGA gomega(alps::gf::matsubara_positive_mesh(beta, n_tau),
                alps::gf::index_mesh(n_flavors),
                alps::gf::index_mesh(n_flavors));
  for (int flavor = 0; flavor < n_flavors; ++flavor) {
    for (int flavor2 = 0; flavor2 < n_flavors; ++flavor2) {
      for (int il = 0; il < n_legendre; ++il) {
        tmp_mat(il, 0) = Gl[flavor][flavor2][il];
      }
      tmp_mat2 = Tnl * tmp_mat;
      for (int im = 0; im < n_tau; ++im) {
        gomega(g::matsubara_index(im), g::index(flavor), g::index(flavor2)) = tmp_mat2(im, 0) / sign;
      }
    }
  }
  gomega.save(ar, "/gf");

}
