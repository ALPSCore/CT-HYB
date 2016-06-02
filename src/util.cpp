#include "util.hpp"

#include <vector>

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
bool my_isnan(boost::multiprecision::cpp_dec_float_50 x) {
    return boost::math::isnan(x);
}

template<>
bool my_isnan(std::complex<double> x) {
    return my_isnan(get_real(x)) || my_isnan(get_imag(x));
}

double permutation(size_t N, size_t k) {
    assert(k>0);
    double r=1.0;
    for(size_t i=N-k+1; i<=N; ++i) {
        r *= static_cast<double>(i);
    }
    return r;
}

double mymod(double x, double beta) {
    if (x>=0) {
        return x-beta*static_cast<int>(x/beta);
    } else {
        return x+beta*(static_cast<int>(-x/beta)+1);
    }
}
