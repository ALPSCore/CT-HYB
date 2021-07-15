#pragma once

#include <complex>
#include <unordered_map>

#include <Eigen/Dense>
#include <Eigen/LU>

#include <alps/gf/gf.hpp>
#include <alps/gf/tail.hpp>

#include "hash.hpp"
#include "common/util.hpp"


template<typename SOLVER_TYPE>
void compute_two_time_G2(const typename alps::results_type<SOLVER_TYPE>::type &results,
                         const typename alps::parameters_type<SOLVER_TYPE>::type &parms,
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

  ar["TWO_TIME_G2_LEGENDRE"] = data;
}


inline double tl1(int l) {
  return l%2==0 ? -2 * std::sqrt(2.*l+1.) : 0.0;
}

/// Correct the first moment of asymptotic behavior c/iwn.
/// c=1 for diagonal components and c=0 for off-diaognal components.
template<typename T>
void correct_G1_tail(double beta, boost::multi_array<T,3>& Gl) {
  int n_flavors = Gl.shape()[0];
  int nl = Gl.shape()[2];

  for (int f1 = 0; f1 < n_flavors; ++f1) {
    for (int f2 = 0; f2 < n_flavors; ++f2) {
      double c1 = (f1 == f2 ? 1 : 0);

      T diff = beta * c1;
      T sum_t2 = 0.0;
      for (int l=0; l<nl; ++l) {
        diff -= tl1(l) * Gl[f1][f2][l];
        sum_t2 += tl1(l) * tl1(l);
      }

      for (int l=0; l<nl; ++l) {
        Gl[f1][f2][l] += (diff/sum_t2) * tl1(l);
      }
    }
  }
}

template<typename SOLVER_TYPE>
void compute_G1(const typename alps::results_type<SOLVER_TYPE>::type &results,
                const typename alps::parameters_type<SOLVER_TYPE>::type &parms,
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

  const std::vector<double> Gl_Re = results["G1_Re"].template mean<std::vector<double> >();
  const std::vector<double> Gl_Im = results["G1_Im"].template mean<std::vector<double> >();
  assert(Gl_Re.size() == n_flavors * n_flavors * n_legendre);
  boost::multi_array<std::complex<double>, 3>
      Gl(boost::extents[n_flavors][n_flavors][n_legendre]);
  std::transform(Gl_Re.begin(), Gl_Re.end(), Gl_Im.begin(), Gl.origin(), to_complex<double>());
  std::transform(Gl.origin(), Gl.origin() + Gl.num_elements(), Gl.origin(),
                 std::bind1st(std::multiplies<std::complex<double> >(), coeff));

  //correct the first moment
  correct_G1_tail(beta, Gl);
  ar["G1_LEGENDRE"] = Gl;

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
              Pvals[il] * Gl[flavor][flavor2][il] * sqrt_array[il] * temperature;
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
        tmp_mat(il, 0) = Gl[flavor][flavor2][il];
      }
      tmp_mat2 = Tnl * tmp_mat;
      for (int im = 0; im < n_matsubara; ++im) {
        gomega(g::matsubara_index(im), g::index(flavor), g::index(flavor2)) = tmp_mat2(im, 0);
      }
    }
  }
  ar["gf"] = gomega;
}

template<typename SOLVER_TYPE>
void compute_G2_matsubara(const typename alps::results_type<SOLVER_TYPE>::type &results,
                const typename alps::parameters_type<SOLVER_TYPE>::type &parms,
                bool verbose = false) {

  const std::string output_file = parms["outputfile"];

  auto freqs = read_matsubara_points(parms["measurement.G2.matsubara.frequencies_PH"]);
  const int n_flavors = parms["model.sites"].template as<int>() * parms["model.spins"].template as<int>();
  const double sign = results["Sign"].template mean<double>();

  //The factor of temperature below comes from the extra degree of freedom for beta in the worm
  const double coeff =
      results["worm_space_volume_G2"].template mean<double>() /
          (sign * results["Z_function_space_volume"].template mean<double>());


  boost::multi_array<std::complex<double>, 5> G2iwn_H;
  boost::multi_array<int,2> freqs_meas;
  alps::hdf5::archive ar(output_file, "rw");
  ar["/simulation/results/G2H_matsubara/data"] >> G2iwn_H;
  ar["/simulation/results/G2H_matsubara/freqs_PH"] >> freqs_meas;
  std::transform(G2iwn_H.origin(), G2iwn_H.origin() + G2iwn_H.num_elements(), G2iwn_H.origin(),
                 std::bind1st(std::multiplies<std::complex<double> >(), coeff));

  // Loop over frequencies actually measured
  std::unordered_map<std::tuple<int,int,int>, int> freqs_map;
  for (int i=0; i<freqs_meas.size(); ++i) {
    int freq_f1 = freqs_meas[i][0];
    int freq_f2 = freqs_meas[i][1];
    int freq_b = freqs_meas[i][2];
    freqs_map[matsubara_freq_point_PH(freq_f1,freq_f2,freq_b)] = i;
  }

  // Since we measured only the Hartree term, we have to recover the contribution of the Fock term.
  boost::multi_array<std::complex<double>, 5>
      G2iwn(boost::extents[n_flavors][n_flavors][n_flavors][n_flavors][freqs.size()]);

  // Loop over frequencies that are requested to measure by the user
  boost::multi_array<int,2> freqs_array(boost::extents[freqs.size()][3]);
  for (int ifreq=0; ifreq<freqs.size(); ++ifreq) {
    int freq_f1 = std::get<0>(freqs[ifreq]);
    int freq_f2 = std::get<1>(freqs[ifreq]);
    int freq_b = std::get<2>(freqs[ifreq]);

    freqs_array[ifreq][0] = freq_f1;
    freqs_array[ifreq][1] = freq_f2;
    freqs_array[ifreq][2] = freq_b;

    auto freq_H = matsubara_freq_point_PH(freq_f1, freq_f2, freq_b);
    auto freq_F = from_H_to_F(freq_H);
    if (freqs_map.find(freq_H) == freqs_map.end()) {
       std::cerr << "Not found: "
           << std::get<0>(freq_H) << " "
           << std::get<1>(freq_H) << " "
           << std::get<2>(freq_H) << std::endl;
    }
    auto ifreq_H = freqs_map.at(freq_H);
    auto ifreq_F = freqs_map.at(freq_F);

    for (int f1 = 0; f1 < n_flavors; ++f1) {
      for (int f2 = 0; f2 < n_flavors; ++f2) {
        for (int f3 = 0; f3 < n_flavors; ++f3) {
          for (int f4 = 0; f4 < n_flavors; ++f4) {
            G2iwn[f1][f2][f3][f4][ifreq] = G2iwn_H[ifreq_H][f1][f2][f3][f4] - G2iwn_H[ifreq_F][f1][f4][f3][f2];
          }
        }
      }
    }
  }

  ar["/G2/matsubara/freqs_PH"] << freqs_array;
  ar["/G2/matsubara/data"] << G2iwn;
}

template<typename SOLVER_TYPE>
void compute_G2(const typename alps::results_type<SOLVER_TYPE>::type &results,
                const typename alps::parameters_type<SOLVER_TYPE>::type &parms,
                std::map<std::string,boost::any> &ar,
                bool verbose = false) {
  const int n_legendre(parms["measurement.G2.legendre.n_legendre"]);
  const int n_freq(parms["measurement.G2.legendre.n_bosonic_freq"]);
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

  ar["G2_LEGENDRE"] = Gl;
}

template<typename SOLVER_TYPE>
void compute_euqal_time_G1(const typename alps::results_type<SOLVER_TYPE>::type &results,
                           const typename alps::parameters_type<SOLVER_TYPE>::type &parms,
                           std::map<std::string,boost::any> &ar,
                           bool verbose = false) {
  const double beta(parms["model.beta"]);
  const int n_flavors = parms["model.sites"].template as<int>() * parms["model.spins"].template as<int>();
  const double temperature(1.0 / beta);
  const double sign = results["Sign"].template mean<double>();
  const double coeff =
      temperature * results["worm_space_volume_Equal_time_G1"].template mean<double>() /
          (sign * results["Z_function_space_volume"].template mean<double>());

  const std::vector<double> data_Re = results["Equal_time_G1_Re"].template mean<std::vector<double> >();
  const std::vector<double> data_Im = results["Equal_time_G1_Im"].template mean<std::vector<double> >();
  assert(data_Re.size() == n_flavors * n_flavors);
  boost::multi_array<std::complex<double>, 2> data(boost::extents[n_flavors][n_flavors]);
  std::transform(data_Re.begin(), data_Re.end(), data_Im.begin(), data.origin(), to_complex<double>());
  std::transform(data.origin(), data.origin() + data.num_elements(), data.origin(),
                 std::bind1st(std::multiplies<std::complex<double> >(), coeff));
  ar["EQUAL_TIME_G1"] = data;
}

template<typename SOLVER_TYPE>
void compute_equal_time_G2(const typename alps::results_type<SOLVER_TYPE>::type &results,
                           const typename alps::parameters_type<SOLVER_TYPE>::type &parms,
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

  const std::vector<double> data_Re = results["Equal_time_G2_Re"].template mean<std::vector<double> >();
  const std::vector<double> data_Im = results["Equal_time_G2_Im"].template mean<std::vector<double> >();
  assert(data_Re.size() == n_flavors * n_flavors * n_flavors * n_flavors);
  boost::multi_array<std::complex<double>, 4>
      data(boost::extents[n_flavors][n_flavors][n_flavors][n_flavors]);
  std::transform(data_Re.begin(), data_Re.end(), data_Im.begin(), data.origin(), to_complex<double>());
  std::transform(data.origin(), data.origin() + data.num_elements(), data.origin(),
                 std::bind1st(std::multiplies<std::complex<double> >(), coeff));
  ar["EQUAL_TIME_G2"] = data;
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