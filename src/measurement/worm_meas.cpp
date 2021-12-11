#include "worm_meas.ipp"

typedef SlidingWindowManager<REAL_EIGEN_BASIS_MODEL> SW_REAL_MATRIX;
typedef SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL> SW_COMPLEX_MATRIX;

template class WormConfigRecord<double,double>;


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
