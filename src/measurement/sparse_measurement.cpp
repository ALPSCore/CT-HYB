#include "sparse_measurement.hpp"
#include "sparse_measurement.ipp"

typedef SlidingWindowManager<REAL_EIGEN_BASIS_MODEL> SW_REAL_MATRIX;
typedef SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL> SW_COMPLEX_MATRIX;

void init_work_space(boost::multi_array<std::complex<double>, 3> &data, int num_flavors, int num_legendre, int num_freq) {
  data.resize(boost::extents[num_flavors][num_flavors][num_legendre]);
  std::fill(data.origin(), data.origin() + data.num_elements(), 0.0);
}

void init_work_space(boost::multi_array<std::complex<double>, 7> &data, int num_flavors, int num_legendre, int num_freq) {
  data.resize(boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][num_legendre][num_legendre][num_freq]);
  std::fill(data.origin(), data.origin() + data.num_elements(), 0.0);
}

inline int to_old_convention(int i) {
  if (i%2 == 0) {
    //boson
    return i/2;
  } else {
    //fermion
    if (i>=0) {
      return i/2;
    } else {
      return i/2-1;
    }
  }
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
      throw std::runtime_error("The first column has a wrong value in " + file + ".");
    }
    data[i] = matsubara_freq_point_PH(
      to_old_convention(n),
      to_old_convention(np),
      to_old_convention(m)
    );
  }

  return data;
}


// make a list of fermionic frequencies of one-particle-GF-like object
void make_two_freqs_list(
    const std::vector<matsubara_freq_point_PH>& freqs_PH,
    std::vector<std::pair<int,int>>& two_freqs_vec,
    std::unordered_map<std::pair<int,int>, int>& two_freqs_map) {

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
  for (auto& freq_PH: freqs_PH) {
    auto freq_f1 = std::get<0>(freq_PH);
    auto freq_f2 = std::get<1>(freq_PH);
    auto freq_b = std::get<2>(freq_PH);
    add(freq_f1+freq_b, freq_f1);
    add(freq_f2, freq_f2+freq_b);
  }

  two_freqs_map.clear();
  for (int f = 0; f < two_freqs_vec.size(); ++f) {
    two_freqs_map.emplace(two_freqs_vec[f], f);
  }
}

template class G2SparseMeasurement<double, SW_REAL_MATRIX>;
template class G2SparseMeasurement<std::complex<double>, SW_COMPLEX_MATRIX>;