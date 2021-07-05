#include "util.hpp"

#include <vector>

double myabs(double x) {
  return std::abs(x);
}

double myabs(std::complex<double> x) {
  return std::abs(x);
}

double mymod(double x, double beta) {
  if (x >= 0) {
    return x - beta * static_cast<int>(x / beta);
  } else {
    return x + beta * (static_cast<int>(-x / beta) + 1);
  }
}

double get_real(std::complex<double> x) {
  return x.real();
}

double get_imag(std::complex<double> x) {
  return x.imag();
}

double get_real(double x) {
  return x;
}

double get_imag(double x) {
  return 0.0;
}

double min_distance(double dist, double BETA) {
  const double abs_dist = std::abs(dist);
  assert(abs_dist >= 0 && abs_dist <= BETA);
  return std::min(abs_dist, BETA - abs_dist);
}

template<>
double mycast(std::complex<double> val) {
  return val.real();
}

template<>
std::complex<double> mycast(std::complex<double> val) {
  return val;
}

template<>
double myconj(double val) {
  return val;
}

template<>
std::complex<double> myconj(std::complex<double> val) {
  return std::conj(val);
}

template<>
bool my_isnan(double x) {
  return std::isnan(x);
}

template<>
bool my_isnan(std::complex<double> x) {
  return my_isnan(get_real(x)) || my_isnan(get_imag(x));
}


double permutation(size_t N, size_t k) {
  assert(k > 0);
  double r = 1.0;
  for (size_t i = N - k + 1; i <= N; ++i) {
    r *= static_cast<double>(i);
  }
  return r;
}


