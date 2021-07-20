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


std::vector<int> read_fermionic_matsubara_points(const std::string& file) {
  std::ifstream f(file);

  if (!f.is_open()) {
    throw std::runtime_error("File at " + file + ", which should contain fermionic frequencies cannot be read or does not exit.");
  }

  int num_freqs;
  f >> num_freqs;

  std::vector<int> data(num_freqs);
  for (int i=0; i<num_freqs; ++i) {
    int j, n;
    f >> j >> n;
    if (i != j) {
      throw std::runtime_error("The first column has a wrong value in " + file + ".");
    }
    data[i] = n;
  }

  check_true(is_fermionic(data.begin(), data.end()), "Some of frequencies are not fermionic (odd integer)!");

  return data;
}
