#include "worm_meas.ipp"

typedef SlidingWindowManager<REAL_EIGEN_BASIS_MODEL> SW_REAL_MATRIX;
typedef SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL> SW_COMPLEX_MATRIX;

template class EqualTimeG1Meas<double,SW_REAL_MATRIX>;
template class EqualTimeG1Meas<std::complex<double>,SW_COMPLEX_MATRIX>;

void compute_equal_time_G1(
    const alps::accumulators::result_set &results,
    int n_flavors,
    double beta,
    double sign,
    double worm_space_rel_vol,
    std::map<std::string,boost::any> &ar,
    bool verbose) {
  std::vector<double> data_Re = results["Equal_time_G1_Re"].template mean<std::vector<double> >();
  std::vector<double> data_Im = results["Equal_time_G1_Im"].template mean<std::vector<double> >();
  double coeff = worm_space_rel_vol/(sign * beta);

  check_true(data_Re.size() == n_flavors * n_flavors);
  boost::multi_array<std::complex<double>, 2> data(boost::extents[n_flavors][n_flavors]);
  std::transform(data_Re.begin(), data_Re.end(), data_Im.begin(), data.origin(), to_complex<double>());
  std::transform(data.origin(), data.origin() + data.num_elements(), data.origin(),
                 std::bind1st(std::multiplies<std::complex<double> >(), coeff));
  ar["EQUAL_TIME_G1"] = data;
}