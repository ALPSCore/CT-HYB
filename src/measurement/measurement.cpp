#include "measurement.hpp"
#include "measurement.ipp"

typedef SlidingWindowManager<REAL_EIGEN_BASIS_MODEL> SW_REAL_MATRIX;
typedef SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL> SW_COMPLEX_MATRIX;

void init_work_space(boost::multi_array<std::complex<double>, 3> &data, int num_flavors, int num_legendre, int num_freq) {
  data.resize(boost::extents[num_flavors][num_flavors][num_legendre]);
}

void init_work_space(boost::multi_array<std::complex<double>, 7> &data, int num_flavors, int num_legendre, int num_freq) {
  data.resize(boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][num_legendre][num_legendre][num_freq]);
}

std::vector<matsubara_freq_point_PH> read_matsubara_points(const std::string& file) {
  std::ifstream f(file);

  if (!f.is_open()) {
    throw std::runtime_error("File at " + file + ", which should contain the list of Matsubara frequencies for G2 measurement cannot be read or does not exit.");
  }

  int num_freqs;
  f >> num_freqs;

  std::vector<matsubara_freq_point_PH> data(num_freqs);
  for (int i=0; i<num_freqs; ++i) {
    int j, n, np, m;
    f >> j >> n >> np >> m;
    if (i != j) {
      std::runtime_error("The first column has a wrong value in " + file + ".");
    }
    data[i] = matsubara_freq_point_PH(n, np, m);
  }

  return data;
}

#undef PP_SCALAR
#undef PP_EXTENDED_SCALAR
#undef PP_SW
#define PP_SCALAR double
#define PP_EXTENDED_SCALAR EXTENDED_REAL
#define PP_SW SW_REAL_MATRIX
#include "measurement_explicit.def"

#undef PP_SCALAR
#undef PP_EXTENDED_SCALAR
#undef PP_SW
#define PP_SCALAR std::complex<double>
#define PP_EXTENDED_SCALAR EXTENDED_COMPLEX
#define PP_SW SW_COMPLEX_MATRIX
#include "measurement_explicit.def"
