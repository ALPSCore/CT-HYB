#include "vartheta.ipp"

typedef SlidingWindowManager<REAL_EIGEN_BASIS_MODEL> SW_REAL_MATRIX;
typedef SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL> SW_COMPLEX_MATRIX;

template class VarThetaMeas<double,SW_REAL_MATRIX>;
template class VarThetaMeas<std::complex<double>,SW_COMPLEX_MATRIX>;

template class VarThetaLegendreMeas<double,SW_REAL_MATRIX>;
template class VarThetaLegendreMeas<std::complex<double>,SW_COMPLEX_MATRIX>;

void compute_vartheta_legendre(
    const alps::accumulators::result_set &results,
    int n_flavors,
    double beta,
    double sign,
    double worm_space_rel_vol,
    std::map<std::string,boost::any> &ar,
    bool verbose) {
  std::vector<double> data_Re = results["vartheta_legendre_Re"].template mean<std::vector<double> >();
  std::vector<double> data_Im = results["vartheta_legendre_Im"].template mean<std::vector<double> >();
  double coeff = worm_space_rel_vol/(sign * beta);

  check_true(data_Re.size() % n_flavors * n_flavors == 0);
  int nl = data_Re.size()/(n_flavors * n_flavors);

  boost::multi_array<std::complex<double>,3> data(boost::extents[nl][n_flavors][n_flavors]);
  std::transform(data_Re.begin(), data_Re.end(), data_Im.begin(), data.origin(), to_complex<double>());
  std::transform(data.origin(), data.origin() + data.num_elements(), data.origin(),
                [&](const auto&x ){return -coeff*x;});
  ar["vartheta_legendre"] = data;
}

void compute_vartheta(
    const alps::accumulators::result_set &results,
    int n_flavors,
    double beta,
    double sign,
    double worm_space_rel_vol,
    std::map<std::string,boost::any> &ar,
    bool verbose) {
  std::vector<double> data_Re = results["vartheta_Re"].template mean<std::vector<double> >();
  std::vector<double> data_Im = results["vartheta_Im"].template mean<std::vector<double> >();
  double coeff = worm_space_rel_vol/(sign * beta);

  check_true(data_Re.size() % n_flavors * n_flavors == 0);
  int nvsample = data_Re.size()/(n_flavors * n_flavors);

  boost::multi_array<std::complex<double>,3> data(boost::extents[nvsample][n_flavors][n_flavors]);
  std::transform(data_Re.begin(), data_Re.end(), data_Im.begin(), data.origin(), to_complex<double>());
  std::transform(data.origin(), data.origin() + data.num_elements(), data.origin(),
                [&](const auto&x ){return -coeff*x;});
  ar["vartheta"] = data;
}