#pragma once

#include <complex>

#include <Eigen/Dense>
#include <Eigen/LU>

#include <alps/gf/gf.hpp>
#include <alps/gf/tail.hpp>

#include "src/orthogonal_basis/basis.hpp"

//template<typename T>
//struct to_complex {
  //std::complex<T> operator()(const T &re, const T &im) {
    //return std::complex<T>(re, im);
  //}
//};

template<typename SOLVER_TYPE>
void compute_two_time_G2(const typename alps::results_type<SOLVER_TYPE>::type &results,
                         const typename alps::parameters_type<SOLVER_TYPE>::type &parms,
                         const Eigen::Matrix<typename SOLVER_TYPE::SCALAR,
                                             Eigen::Dynamic,
                                             Eigen::Dynamic> &rotmat_Delta,
                         std::map<std::string,boost::any> &ar,
                         bool verbose = false) {
  const int dim_ir_basis(parms["measurement.two_time_G2.dim_ir_basis"].template as<int>());
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
  assert(data_Re.size() == n_flavors * n_flavors * n_flavors * n_flavors * dim_ir_basis);
  boost::multi_array<std::complex<double>, 5>
      data(boost::extents[n_flavors][n_flavors][n_flavors][n_flavors][dim_ir_basis]);
  std::transform(data_Re.begin(), data_Re.end(), data_Im.begin(), data.origin(), to_complex<double>());
  std::transform(data.origin(), data.origin() + data.num_elements(), data.origin(),
                 std::bind1st(std::multiplies<std::complex<double> >(), coeff));

  boost::multi_array<std::complex<double>, 5>
      data_org_basis(boost::extents[n_flavors][n_flavors][n_flavors][n_flavors][dim_ir_basis]);
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
                  for (int il = 0; il < dim_ir_basis; ++il) {
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
