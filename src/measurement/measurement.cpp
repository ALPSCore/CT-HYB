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


// make a list of fermionic frequencies of one-particle-GF-like object
void make_two_freqs_list(
    const std::vector<matsubara_freq_point_PH>& freqs_PH,
    std::vector<std::pair<int,int>>& two_freqs_vec,
    std::unordered_map<std::pair<int,int>, int, HashIntPair>& two_freqs_map) {

  // two_freqs_vec contains a list of fermionic frequencies of one-particle-GF-like object
  std::set<std::pair<int,int>> two_freqs_set;
  two_freqs_vec.resize(0);
  auto add = [&](int f1, int f2) {
    auto key = std::pair<int,int>(f1, f2);
    if (two_freqs_set.find(key) == two_freqs_set.end()) {
      two_freqs_vec.push_back(key);
      two_freqs_set.insert(key);
    };
  };
  auto add_PH = [&](const matsubara_freq_point_PH& freq_PH) {
    auto freq_f1 = std::get<0>(freq_PH);
    auto freq_f2 = std::get<1>(freq_PH);
    auto freq_b = std::get<2>(freq_PH);
    add(freq_f1+freq_b, freq_f1);
    add(freq_f2, freq_f2+freq_b);
  };
  for (auto& freq_PH: freqs_PH) {
    // For measuring Hartree term
    add_PH(freq_PH);

    // For measuring Fock term
    auto freq_f1 = std::get<0>(freq_PH);
    auto freq_f2 = std::get<1>(freq_PH);
    auto freq_b = std::get<2>(freq_PH);
    matsubara_freq_point_PH freq_PH_F;
    std::get<0>(freq_PH_F) = freq_f2 + freq_b;
    std::get<1>(freq_PH_F) = freq_f2;
    std::get<2>(freq_PH_F) = freq_f1 - freq_f2;
    add_PH(freq_PH_F);
    //add(freq_f1+freq_b, freq_f1);
    //add(freq_f2, freq_f2+freq_b);
    //add(freq_f1+freq_b, freq_f2+freq_b);
    //add(freq_f2, freq_f1);
  }

  two_freqs_map.clear();
  for (int f = 0; f < two_freqs_vec.size(); ++f) {
    two_freqs_map.emplace(two_freqs_vec[f], f);
  }
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
