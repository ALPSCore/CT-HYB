#pragma once

#include <complex>

#include <Eigen/Dense>
#include <Eigen/LU>

#include <alps/gf/gf.hpp>
#include <alps/gf/tail.hpp>

#include "legendre.hpp"

template<typename T>
struct to_complex {
  std::complex<T> operator()(const T &re, const T &im) {
    return std::complex<T>(re, im);
  }
};

template<typename SOLVER_TYPE>
void compute_two_time_G2(const typename alps::results_type<SOLVER_TYPE>::type &results,
                         const typename alps::parameters_type<SOLVER_TYPE>::type &parms,
                         const Eigen::Matrix<typename SOLVER_TYPE::SCALAR,
                                             Eigen::Dynamic,
                                             Eigen::Dynamic> &rotmat_Delta,
                         std::map<std::string,boost::any> &ar,
                         bool verbose = false) {
  const int n_legendre(parms["measurement.two_time_G2.n_legendre"].template as<int>());
  const double beta(parms["model.beta"]);
  const int n_flavors = parms["model.sites"].template as<int>() * parms["model.spins"].template as<int>();
  const double temperature(1.0 / beta);
  const double coeff =
      temperature * results["worm_space_volume_Two_time_G2"].template mean<double>() /
          (results["Sign"].template mean<double>() * results["Z_function_space_volume"].template mean<double>());

  if (verbose) {
    std::cout << "Volumes of Two_time_G2 space/Z_function_space is "
              << results["worm_space_volume_Two_time_G2"].template mean<double>()
              << " : " << results["Z_function_space_volume"].template mean<double>() << std::endl;
  }

  const std::vector<double> data_Re = results["Two_time_G2_Re"].template mean<std::vector<double> >();
  const std::vector<double> data_Im = results["Two_time_G2_Im"].template mean<std::vector<double> >();
  assert(data_Re.size() == n_flavors * n_flavors * n_flavors * n_flavors * n_legendre);
  boost::multi_array<std::complex<double>, 5>
      data(boost::extents[n_flavors][n_flavors][n_flavors][n_flavors][n_legendre]);
  std::transform(data_Re.begin(), data_Re.end(), data_Im.begin(), data.origin(), to_complex<double>());
  std::transform(data.origin(), data.origin() + data.num_elements(), data.origin(),
                 std::bind1st(std::multiplies<std::complex<double> >(), coeff));

  boost::multi_array<std::complex<double>, 5>
      data_org_basis(boost::extents[n_flavors][n_flavors][n_flavors][n_flavors][n_legendre]);
  std::fill(data_org_basis.origin(), data_org_basis.origin() + data_org_basis.num_elements(), 0.0);

  //basis rotation very ugly. TO DO: replace the loops with tensordots.
  for (int f0 = 0; f0 < n_flavors; ++f0) {
    for (int f1 = 0; f1 < n_flavors; ++f1) {
      for (int f2 = 0; f2 < n_flavors; ++f2) {
        for (int f3 = 0; f3 < n_flavors; ++f3) {
          for (int g0 = 0; g0 < n_flavors; ++g0) {
            for (int g1 = 0; g1 < n_flavors; ++g1) {
              for (int g2 = 0; g2 < n_flavors; ++g2) {
                for (int g3 = 0; g3 < n_flavors; ++g3) {
                  for (int il = 0; il < n_legendre; ++il) {
                    data_org_basis[f0][f1][f2][f3][il] += data[g0][g1][g2][g3][il] *
                        myconj(rotmat_Delta(f0, g0)) *
                        rotmat_Delta(f1, g1) *
                        myconj(rotmat_Delta(f2, g2)) *
                        rotmat_Delta(f3, g3);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  ar["TWO_TIME_G2_LEGENDRE"] = data_org_basis;
}


template<typename SOLVER_TYPE>
void compute_G1(const typename alps::results_type<SOLVER_TYPE>::type &results,
                const typename alps::parameters_type<SOLVER_TYPE>::type &parms,
                const Eigen::Matrix<typename SOLVER_TYPE::SCALAR, Eigen::Dynamic, Eigen::Dynamic> &rotmat_Delta,
                std::map<std::string,boost::any> &ar,
                bool verbose = false) {
  namespace g=alps::gf;
  typedef Eigen::Matrix<typename SOLVER_TYPE::SCALAR, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> complex_matrix_t;

  const int n_legendre(parms["measurement.G1.n_legendre"].template as<int>());
  const int n_tau(parms["measurement.G1.n_tau"]);
  const int n_matsubara(parms["measurement.G1.n_matsubara"]);
  const double beta(parms["model.beta"]);
  const int n_flavors = parms["model.sites"].template as<int>() * parms["model.spins"].template as<int>();
  const double temperature(1.0 / beta);
  const double sign = results["Sign"].template mean<double>();
  //The factor of temperature below comes from the extra degree of freedom for beta in the worm
  const double coeff =
      results["worm_space_volume_G1"].template mean<double>() /
          (sign * results["Z_function_space_volume"].template mean<double>());

  boost::multi_array<std::complex<double>, 3>
      Gl_org_basis(boost::extents[n_flavors][n_flavors][n_legendre]);
  {
    const std::vector<double> Gl_Re = results["G1_Re"].template mean<std::vector<double> >();
    const std::vector<double> Gl_Im = results["G1_Im"].template mean<std::vector<double> >();
    assert(Gl_Re.size() == n_flavors * n_flavors * n_legendre);
    boost::multi_array<std::complex<double>, 3>
        Gl(boost::extents[n_flavors][n_flavors][n_legendre]);
    std::transform(Gl_Re.begin(), Gl_Re.end(), Gl_Im.begin(), Gl.origin(), to_complex<double>());
    std::transform(Gl.origin(), Gl.origin() + Gl.num_elements(), Gl.origin(),
                   std::bind1st(std::multiplies<std::complex<double> >(), coeff));

    //rotate back to the original basis
    complex_matrix_t mattmp(n_flavors, n_flavors), mattmp2(n_flavors, n_flavors);
    const matrix_t inv_rotmat_Delta = rotmat_Delta.inverse();
    for (int il = 0; il < n_legendre; ++il) {
      for (int flavor1 = 0; flavor1 < n_flavors; ++flavor1) {
        for (int flavor2 = 0; flavor2 < n_flavors; ++flavor2) {
          mattmp(flavor1, flavor2) = Gl[flavor1][flavor2][il];
        }
      }
      mattmp2 = rotmat_Delta * mattmp * inv_rotmat_Delta;
      for (int flavor1 = 0; flavor1 < n_flavors; ++flavor1) {
        for (int flavor2 = 0; flavor2 < n_flavors; ++flavor2) {
          Gl_org_basis[flavor1][flavor2][il] = mattmp2(flavor1, flavor2);
        }
      }
    }
  }
  ar["G1_LEGENDRE"] = Gl_org_basis;

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
              Pvals[il] * Gl_org_basis[flavor][flavor2][il] * sqrt_array[il] * temperature;
        }
      }
    }
  }
  ar["gtau"] = itime_gf;

  /*
   * Compute Gomega from Legendre coefficients
   */
  typedef alps::gf::three_index_gf<std::complex<double>, alps::gf::matsubara_positive_mesh,
                                   alps::gf::index_mesh,
                                   alps::gf::index_mesh
  > GOMEGA;

  const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &Tnl(legendre_transformer.Tnl());
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> tmp_mat(n_legendre, 1), tmp_mat2(n_matsubara, 1);
  GOMEGA gomega(alps::gf::matsubara_positive_mesh(beta, n_matsubara),
                alps::gf::index_mesh(n_flavors),
                alps::gf::index_mesh(n_flavors));
  for (int flavor = 0; flavor < n_flavors; ++flavor) {
    for (int flavor2 = 0; flavor2 < n_flavors; ++flavor2) {
      for (int il = 0; il < n_legendre; ++il) {
        tmp_mat(il, 0) = Gl_org_basis[flavor][flavor2][il];
      }
      tmp_mat2 = Tnl * tmp_mat;
      for (int im = 0; im < n_matsubara; ++im) {
        gomega(g::matsubara_index(im), g::index(flavor), g::index(flavor2)) = tmp_mat2(im, 0);
      }
    }
  }
  ar["gf"] = gomega;
}

//very crapy way to implement ...
template<typename SCALAR>
void rotate_back_G2(int n_flavors, boost::multi_array<std::complex<double>, 4> &G2,
                    const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> &rotmat_Delta) {

  boost::multi_array<std::complex<double>, 4>
      work(boost::extents[n_flavors][n_flavors][n_flavors][n_flavors]);

  //transform the first index
  std::fill(work.origin(), work.origin() + work.num_elements(), 0.0);
  for (int f1 = 0; f1 < n_flavors; ++f1) {
    for (int f2 = 0; f2 < n_flavors; ++f2) {
      for (int f3 = 0; f3 < n_flavors; ++f3) {
        for (int f4 = 0; f4 < n_flavors; ++f4) {
          for (int g1 = 0; g1 < n_flavors; ++g1) {
            work[f1][f2][f3][f4] += rotmat_Delta(f1, g1) * G2[g1][f2][f3][f4];
          }
        }
      }
    }
  }
  G2 = work;

  //transform the second index
  std::fill(work.origin(), work.origin() + work.num_elements(), 0.0);
  for (int f1 = 0; f1 < n_flavors; ++f1) {
    for (int f2 = 0; f2 < n_flavors; ++f2) {
      for (int f3 = 0; f3 < n_flavors; ++f3) {
        for (int f4 = 0; f4 < n_flavors; ++f4) {
          for (int g2 = 0; g2 < n_flavors; ++g2) {
            work[f1][f2][f3][f4] += myconj(rotmat_Delta(f2, g2)) * G2[f1][g2][f3][f4];
          }
        }
      }
    }
  }
  G2 = work;

  //transform the third index
  std::fill(work.origin(), work.origin() + work.num_elements(), 0.0);
  for (int f1 = 0; f1 < n_flavors; ++f1) {
    for (int f2 = 0; f2 < n_flavors; ++f2) {
      for (int f3 = 0; f3 < n_flavors; ++f3) {
        for (int f4 = 0; f4 < n_flavors; ++f4) {
          for (int g3 = 0; g3 < n_flavors; ++g3) {
            work[f1][f2][f3][f4] += rotmat_Delta(f3, g3) * G2[f1][f2][g3][f4];
          }
        }
      }
    }
  }
  G2 = work;

  //transform the fourth index
  std::fill(work.origin(), work.origin() + work.num_elements(), 0.0);
  for (int f1 = 0; f1 < n_flavors; ++f1) {
    for (int f2 = 0; f2 < n_flavors; ++f2) {
      for (int f3 = 0; f3 < n_flavors; ++f3) {
        for (int f4 = 0; f4 < n_flavors; ++f4) {
          for (int g4 = 0; g4 < n_flavors; ++g4) {
            work[f1][f2][f3][f4] += myconj(rotmat_Delta(f4, g4)) * G2[f1][f2][f3][g4];
          }
        }
      }
    }
  }
  G2 = work;
}

template<typename SOLVER_TYPE>
void compute_G2(const typename alps::results_type<SOLVER_TYPE>::type &results,
                const typename alps::parameters_type<SOLVER_TYPE>::type &parms,
                const Eigen::Matrix<typename SOLVER_TYPE::SCALAR, Eigen::Dynamic, Eigen::Dynamic> &rotmat_Delta,
                std::map<std::string,boost::any> &ar,
                bool verbose = false) {
  const int n_legendre(parms["measurement.G2.n_legendre"]);
  const int n_freq(parms["measurement.G2.n_bosonic_freq"]);
  const int n_flavors = parms["model.sites"].template as<int>() * parms["model.spins"].template as<int>();
  const double sign = results["Sign"].template mean<double>();

  //The factor of temperature below comes from the extra degree of freedom for beta in the worm
  const double coeff =
      results["worm_space_volume_G2"].template mean<double>() /
          (sign * results["Z_function_space_volume"].template mean<double>());

  const std::vector<double> Gl_Re = results["G2_Re"].template mean<std::vector<double> >();
  const std::vector<double> Gl_Im = results["G2_Im"].template mean<std::vector<double> >();
  boost::multi_array<std::complex<double>, 7>
      Gl(boost::extents[n_flavors][n_flavors][n_flavors][n_flavors][n_legendre][n_legendre][n_freq]);
  std::transform(Gl_Re.begin(), Gl_Re.end(), Gl_Im.begin(), Gl.origin(), to_complex<double>());
  std::transform(Gl.origin(), Gl.origin() + Gl.num_elements(), Gl.origin(),
                 std::bind1st(std::multiplies<std::complex<double> >(), coeff));

  //rotate back to the original basis (using not-cache-friendly approach...)
  for (int il = 0; il < n_legendre; ++il) {
    for (int il2 = 0; il2 < n_legendre; ++il2) {
      for (int ifreq = 0; ifreq < n_freq; ++ifreq) {

        //copy data to work1
        boost::multi_array<std::complex<double>, 4>
            work(boost::extents[n_flavors][n_flavors][n_flavors][n_flavors]);
        for (int f1 = 0; f1 < n_flavors; ++f1) {
          for (int f2 = 0; f2 < n_flavors; ++f2) {
            for (int f3 = 0; f3 < n_flavors; ++f3) {
              for (int f4 = 0; f4 < n_flavors; ++f4) {
                work[f1][f2][f3][f4] = Gl[f1][f2][f3][f4][il][il2][ifreq];
              }
            }
          }
        }

        rotate_back_G2(n_flavors, work, rotmat_Delta);

        //copy result to Gl
        for (int f1 = 0; f1 < n_flavors; ++f1) {
          for (int f2 = 0; f2 < n_flavors; ++f2) {
            for (int f3 = 0; f3 < n_flavors; ++f3) {
              for (int f4 = 0; f4 < n_flavors; ++f4) {
                Gl[f1][f2][f3][f4][il][il2][ifreq] = work[f1][f2][f3][f4];
              }
            }
          }
        }

      }
    }
  }

  ar["G2_LEGENDRE"] = Gl;
}

template<typename SOLVER_TYPE>
void compute_euqal_time_G1(const typename alps::results_type<SOLVER_TYPE>::type &results,
                           const typename alps::parameters_type<SOLVER_TYPE>::type &parms,
                           const Eigen::Matrix<typename SOLVER_TYPE::SCALAR,
                                               Eigen::Dynamic,
                                               Eigen::Dynamic> &rotmat_Delta,
                           std::map<std::string,boost::any> &ar,
                           bool verbose = false) {
  const double beta(parms["model.beta"]);
  const int n_flavors = parms["model.sites"].template as<int>() * parms["model.spins"].template as<int>();
  const double temperature(1.0 / beta);
  const double sign = results["Sign"].template mean<double>();
  const double coeff =
      temperature * results["worm_space_volume_Equal_time_G1"].template mean<double>() /
          (sign * results["Z_function_space_volume"].template mean<double>());

  boost::multi_array<std::complex<double>, 2> data_org_basis(boost::extents[n_flavors][n_flavors]);
  std::fill(data_org_basis.origin(), data_org_basis.origin() + data_org_basis.num_elements(), 0.0);
  {
    const std::vector<double> data_Re = results["Equal_time_G1_Re"].template mean<std::vector<double> >();
    const std::vector<double> data_Im = results["Equal_time_G1_Im"].template mean<std::vector<double> >();
    assert(data_Re.size() == n_flavors * n_flavors);
    boost::multi_array<std::complex<double>, 2> data(boost::extents[n_flavors][n_flavors]);
    std::transform(data_Re.begin(), data_Re.end(), data_Im.begin(), data.origin(), to_complex<double>());
    std::transform(data.origin(), data.origin() + data.num_elements(), data.origin(),
                   std::bind1st(std::multiplies<std::complex<double> >(), coeff));
    for (int f0 = 0; f0 < n_flavors; ++f0) {
      for (int f1 = 0; f1 < n_flavors; ++f1) {
        for (int g0 = 0; g0 < n_flavors; ++g0) {
          for (int g1 = 0; g1 < n_flavors; ++g1) {
            data_org_basis[f0][f1] += data[g0][g1]
                * myconj(rotmat_Delta(f0, g0)) * rotmat_Delta(f1, g1);
          }
        }
      }
    }
  }
  ar["EQUAL_TIME_G1"] = data_org_basis;
}

template<typename SOLVER_TYPE>
void compute_euqal_time_G2(const typename alps::results_type<SOLVER_TYPE>::type &results,
                           const typename alps::parameters_type<SOLVER_TYPE>::type &parms,
                           const Eigen::Matrix<typename SOLVER_TYPE::SCALAR,
                                               Eigen::Dynamic,
                                               Eigen::Dynamic> &rotmat_Delta,
                           std::map<std::string,boost::any> &ar,
                           bool verbose = false) {
  //typedef Eigen::Matrix<typename SOLVER_TYPE::SCALAR, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
  //typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> complex_matrix_t;
  const double beta(parms["model.beta"]);
  const int n_flavors = parms["model.sites"].template as<int>() * parms["model.spins"].template as<int>();
  const double temperature(1.0 / beta);
  const double sign = results["Sign"].template mean<double>();
  const double coeff =
      temperature * results["worm_space_volume_Equal_time_G2"].template mean<double>() /
          (sign * results["Z_function_space_volume"].template mean<double>());

  boost::multi_array<std::complex<double>, 4>
      data_org_basis(boost::extents[n_flavors][n_flavors][n_flavors][n_flavors]);
  std::fill(data_org_basis.origin(), data_org_basis.origin() + data_org_basis.num_elements(), 0.0);
  {
    const std::vector<double> data_Re = results["Equal_time_G2_Re"].template mean<std::vector<double> >();
    const std::vector<double> data_Im = results["Equal_time_G2_Im"].template mean<std::vector<double> >();
    assert(data_Re.size() == n_flavors * n_flavors * n_flavors * n_flavors);
    boost::multi_array<std::complex<double>, 4>
        data(boost::extents[n_flavors][n_flavors][n_flavors][n_flavors]);
    std::transform(data_Re.begin(), data_Re.end(), data_Im.begin(), data.origin(), to_complex<double>());
    std::transform(data.origin(), data.origin() + data.num_elements(), data.origin(),
                   std::bind1st(std::multiplies<std::complex<double> >(), coeff));
    //const matrix_t inv_rotmat_Delta = rotmat_Delta.inverse();
    for (int f0 = 0; f0 < n_flavors; ++f0) {
      for (int f1 = 0; f1 < n_flavors; ++f1) {
        for (int f2 = 0; f2 < n_flavors; ++f2) {
          for (int f3 = 0; f3 < n_flavors; ++f3) {
            for (int g0 = 0; g0 < n_flavors; ++g0) {
              for (int g1 = 0; g1 < n_flavors; ++g1) {
                for (int g2 = 0; g2 < n_flavors; ++g2) {
                  for (int g3 = 0; g3 < n_flavors; ++g3) {
                    data_org_basis[f0][f1][f2][f3] += data[g0][g1][g2][g3] *
                        myconj(rotmat_Delta(f0, g0)) *
                        rotmat_Delta(f1, g1) *
                        myconj(rotmat_Delta(f2, g2)) *
                        rotmat_Delta(f3, g3);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  ar["EQUAL_TIME_G2"] = data_org_basis;
}

template<unsigned long N>
void load_signed_multi_dimension_data(const alps::accumulators::result_set &results,
                          const std::string &name,
                          boost::multi_array<std::complex<double>, N>& data) {
  std::fill(data.origin(), data.origin() + data.num_elements(), 0.0);
  const std::vector<double> data_Re = results[name + "_Re"].template mean<std::vector<double> >();
  const std::vector<double> data_Im = results[name + "_Im"].template mean<std::vector<double> >();
  if (data_Re.size() != data.num_elements() || data_Im.size() != data.num_elements()) {
    throw std::runtime_error("data size inconsistency in loading observable " + name + "!");
  }
  const std::complex<double> coeff = 1.0 / results["Sign"].template mean<double>();
  std::transform(data_Re.begin(), data_Re.end(), data_Im.begin(), data.origin(), to_complex<double>());
  std::transform(data.origin(), data.origin() + data.num_elements(), data.origin(),
                 std::bind1st(std::multiplies<std::complex<double> >(), coeff));
}

template<typename SOLVER_TYPE>
void compute_nn_corr(const typename alps::results_type<SOLVER_TYPE>::type &results,
                const typename alps::parameters_type<SOLVER_TYPE>::type &parms,
                std::map<std::string,boost::any> &ar) {
  const int n_tau(parms["measurement.nn_corr.n_tau"]);
  const int n_def(parms["measurement.nn_corr.n_def"]);
  const double sign = results["Sign"].template mean<double>();

  boost::multi_array<std::complex<double>, 2> data(boost::extents[n_def][n_tau]);
  load_signed_multi_dimension_data(results, std::string("Two_time_correlation_functions"), data);

  ar["DENSITY_DENSITY_CORRELATION_FUNCTIONS"] = data;
}


template<typename SOLVER_TYPE>
void compute_fidelity_susceptibility(const typename alps::results_type<SOLVER_TYPE>::type &results,
                                     const typename alps::parameters_type<SOLVER_TYPE>::type &parms,
                                     std::map<std::string,boost::any> &ar) {
  std::complex<double> kLkR =
      std::complex<double>(results["kLkR_Re"].template mean<double>(), results["kLkR_Im"].template mean<double>());
  std::complex<double>
      k = std::complex<double>(results["k_Re"].template mean<double>(), results["k_Im"].template mean<double>());
  ar["FIDELITY_SUSCEPTIBILITY"] = (0.5 * (kLkR - 0.25 * k * k)).real();
}
