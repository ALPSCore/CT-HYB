#include "two_point_corr.ipp"

typedef SlidingWindowManager<REAL_EIGEN_BASIS_MODEL> SW_REAL_MATRIX;
typedef SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL> SW_COMPLEX_MATRIX;

template class TwoPointCorrMeas<double,SW_REAL_MATRIX,PH_CHANNEL>;
template class TwoPointCorrMeas<std::complex<double>,SW_COMPLEX_MATRIX,PH_CHANNEL>;

template class TwoPointCorrMeas<double,SW_REAL_MATRIX,PP_CHANNEL>;
template class TwoPointCorrMeas<std::complex<double>,SW_COMPLEX_MATRIX,PP_CHANNEL>;

template<>
std::string
TwoPointCorrMeasGetNameHelper<PH_CHANNEL>::operator()() const {
  return "lambda";
}

template<>
std::string
TwoPointCorrMeasGetNameHelper<PP_CHANNEL>::operator()() const {
  return "varphi";
}

void compute_two_point_corr(
    const std::string& name,
    const alps::accumulators::result_set &results,
    int n_flavors,
    double beta,
    double sign,
    double worm_space_rel_vol,
    std::map<std::string,boost::any> &ar,
    bool verbose) {
  std::vector<double> data_Re = results[name+"_Re"].template mean<std::vector<double> >();
  std::vector<double> data_Im = results[name+"_Im"].template mean<std::vector<double> >();
  double coeff = worm_space_rel_vol/(sign * beta);

  check_true(data_Re.size() % (n_flavors * n_flavors * n_flavors * n_flavors) == 0);
  int data_size = data_Re.size()/(n_flavors * n_flavors * n_flavors * n_flavors);

  boost::multi_array<std::complex<double>,5>
    data(boost::extents[data_size][n_flavors][n_flavors][n_flavors][n_flavors]);
  std::transform(data_Re.begin(), data_Re.end(), data_Im.begin(), data.origin(), to_complex<double>());
  std::transform(data.origin(), data.origin() + data.num_elements(), data.origin(),
                [&](const std::complex<double>&x ){return coeff*x;});
  ar[name] = data;
}
