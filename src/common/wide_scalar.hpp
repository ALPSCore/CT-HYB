//
// Define USE_QUAD_FLAOT to activate the support of quad floats
//  Some of operations will be performed using quad floats.
//

#pragma once

#include <complex>

#ifndef USE_QUAD_PRECISION

typedef double EXTENDED_REAL;
typedef std::complex<double> EXTENDED_COMPLEX;

template<typename T>
double mypow(double x, T N) {
  return std::pow(x, N);
}

inline double convert_to_scalar(const EXTENDED_REAL& x) {
  return x;
}

inline std::complex<double> convert_to_scalar(const EXTENDED_COMPLEX& x) {
  return x;
}

inline double convert_to_double(const double& x) {
  return x;
}

inline double convert_to_double(const std::complex<double>& x) {
  return x.real();
}

inline std::complex<double> convert_to_complex(const double& x) {
  return std::complex<double>(x, 0.0);
}

inline std::complex<double> convert_to_complex(const std::complex<double>& x) {
  return x;
}

#else

#include <boost/multiprecision/cpp_bin_float.hpp>
typedef boost::multiprecision::cpp_bin_float_quad EXTENDED_REAL;

template<typename T>
class wcomplex;
typedef wcomplex<EXTENDED_REAL> EXTENDED_COMPLEX;

template<typename T>
class wcomplex {
 public:
  wcomplex() : a_(0.0), b_(0.0) { };
  wcomplex(T re) : a_(re), b_(0.0) { };
  wcomplex(T re, T im) : a_(re), b_(im) { };
  wcomplex(double re) : a_(re), b_(0.0) { };

  template<typename S>
  wcomplex(const std::complex<S> &cval) : a_(cval.real()), b_(cval.imag()) { };

  T real() const { return a_; }
  T imag() const { return b_; }

  wcomplex<T> &operator*=(const T &re) {
    a_ *= re;
    b_ *= re;
    return *this;
  }

  template<typename S>
  wcomplex<T> &operator*=(const std::complex<S> &z) {
    *this = (*this) * z;
    return *this;
  }

  template<typename S>
  wcomplex<T> &operator+=(const std::complex<S> &z) {
    *this = (*this) + z;
    return *this;
  }

  wcomplex<T> &operator+=(const wcomplex<T> &z) {
    *this = (*this) + z;
    return *this;
  }

  operator std::complex<double>() const {
    return std::complex<double>(a_.template convert_to<double>(), b_.template convert_to<double>());
  }

  template<typename S>
  S convert_to() const {
    return S(a_.template convert_to<double>(), b_.template convert_to<double>());
  };

  double convert_to_double() const {
    return a_.template convert_to<double>();
  };

 private:
  T a_, b_;
};

template<class T>
wcomplex<T>
operator+(const wcomplex<T> &z, const wcomplex<T> &w) {
  T a = z.real();
  T b = z.imag();
  T c = w.real();
  T d = w.imag();
  T x = a + c;
  T y = b + d;
  return wcomplex<T>(x, y);
}

template<class T>
wcomplex<T>
operator+(const wcomplex<T> &z, const std::complex<double> &w) {
  T a = z.real();
  T b = z.imag();
  double c = w.real();
  double d = w.imag();
  T x = a + c;
  T y = b + d;
  return wcomplex<T>(x, y);
}

/*
 * operator*()
 */
template<class T>
wcomplex<T>
operator*(const wcomplex<T> &z, const wcomplex<T> &w) {
  T a = z.real();
  T b = z.imag();
  T c = w.real();
  T d = w.imag();
  T x = a * c - b * d;
  T y = a * d + b * c;
  return wcomplex<T>(x, y);
}

template<typename T, typename S>
wcomplex<T>
operator*(const wcomplex<T> &z, const std::complex<S> &w) {
  T a = z.real();
  T b = z.imag();
  S c = w.real();
  S d = w.imag();
  T x = a * c - b * d;
  T y = a * d + b * c;
  return wcomplex<T>(x, y);
}

template<typename T, typename S>
wcomplex<T>
operator*(const std::complex<S> &z, const wcomplex<T> &w) {
  return w * z;
};

template<typename T>
wcomplex<T>
operator*(const wcomplex<T> &z, double w) {
  T a = z.real();
  T b = z.imag();
  return wcomplex<T>(a * w, b * w);
}

template<typename T>
wcomplex<T>
operator*(double z, const wcomplex<T> &w) {
  return w * z;
}

template<typename T>
wcomplex<T>
operator*(const wcomplex<T> &z, EXTENDED_REAL w) {
  T a = z.real();
  T b = z.imag();
  return wcomplex<T>(a * w, b * w);
}

template<typename T>
wcomplex<T>
operator*(EXTENDED_REAL z, const wcomplex<T> &w) {
  return w * z;
}

/*
 * operator-()
 */
template<class T>
wcomplex<T>
operator-(const wcomplex<T> &z, const wcomplex<T> &w) {
  T a = z.real();
  T b = z.imag();
  T c = w.real();
  T d = w.imag();
  T x = a - c;
  T y = b - d;
  return wcomplex<T>(x, y);
}

/*
 * operator/()
 */
template<typename T>
wcomplex<T>
operator/(const wcomplex<T> &z, const wcomplex<T> &w) {
  T a = z.real();
  T b = z.imag();
  T c = w.real();
  T d = w.imag();
  T coeff = 1.0 / (c * c + d * d);
  T x = (a * c + b * d) * coeff;
  T y = (-a * d + b * c) * coeff;
  return wcomplex<T>(x, y);
}

template<typename T>
wcomplex<T>
operator/(const wcomplex<T> &z, const std::complex<double> &w) {
  T a = z.real();
  T b = z.imag();
  double c = w.real();
  double d = w.imag();
  double coeff = 1.0 / (c * c + d * d);
  T x = (a * c + b * d) * coeff;
  T y = (-a * d + b * c) * coeff;
  return wcomplex<T>(x, y);
}

template<typename T>
wcomplex<T>
operator/(const wcomplex<T> &z, const T &w) {
  return wcomplex<T>(z.real() / w, z.imag() / w);
}

template<typename T>
bool
operator==(const wcomplex<T> &z, const wcomplex<T> &w) {
  return (z.real() == w.real()) && (z.imag() == z.imag());
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const wcomplex<T> &val) {
  os << "( " << val.real() << " , " << val.imag() << " )";
  return os;
}


/**
 * Some auxially functions
 */
//template <unsigned Digits10, class ExponentType2_t, class Allocator = void>
//class cpp_dec_float
inline EXTENDED_REAL get_real(const EXTENDED_REAL &x) {
  return x;
}

inline EXTENDED_REAL get_imag(const EXTENDED_REAL &x) {
  return 0.0;
}

//inline EXTENDED_REAL get_real(const std::complex<EXTENDED_REAL>& x) {
//return x.real();
//}

//inline EXTENDED_REAL get_imag(const std::complex<EXTENDED_REAL>& x) {
//return x.real();
//}

inline EXTENDED_REAL myabs(EXTENDED_REAL x) {
  return boost::multiprecision::abs(x);
}

inline EXTENDED_REAL
myabs(const wcomplex<EXTENDED_REAL> &x) {
  return boost::multiprecision::sqrt(x.real() * x.real() + x.imag() * x.imag());
}

inline
bool my_isnan(EXTENDED_REAL x) {
  return boost::math::isnan(x);
}

inline
bool my_isnan(wcomplex<EXTENDED_REAL> x) {
  return my_isnan(x.real()) || my_isnan(x.imag());
}

template<typename T>
EXTENDED_REAL mypow(EXTENDED_REAL x, T N) {
  return boost::multiprecision::pow(x, N);
}

/*
 * Cast operator
 */
inline double convert_to_scalar(const EXTENDED_REAL &x) {
  return x.convert_to<double>();
}

inline std::complex<double> convert_to_scalar(const EXTENDED_COMPLEX &x) {
  return x.convert_to<std::complex<double> >();
}

inline double convert_to_double(const EXTENDED_REAL &x) {
  return x.convert_to<double>();
}

inline std::complex<double> convert_to_complex(const EXTENDED_REAL &x) {
  return std::complex<double>(x.convert_to<double>(), 0.0);
}

inline std::complex<double> convert_to_complex(const EXTENDED_COMPLEX &x) {
  return std::complex<double>(
      x.real().convert_to<double>(),
      x.imag().convert_to<double>()
  );
}

#endif

//Type traits
template<typename T>
struct ExtendedScalar {
  typedef EXTENDED_REAL value_type;
};

template<typename T>
struct ExtendedScalar<std::complex<T> > {
  typedef EXTENDED_COMPLEX value_type;
};
