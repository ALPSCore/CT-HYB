//
// Simple complex class for boost::multiprecision
// H. Shinaoka
//

#pragma once

#include <complex>

#include <boost/multiprecision/cpp_dec_float.hpp>

namespace alps {
  namespace cthyb {
    template<typename T>
    class wcomplex {
    public:
      wcomplex() : a_(0.0), b_(0.0) {};
      wcomplex(T re) : a_(re), b_(0.0) {};
      wcomplex(T re, T im) : a_(re), b_(im) {};
      wcomplex(double re) : a_(re), b_(0.0) {};

      template<typename S> wcomplex(const std::complex<S>& cval) : a_(cval.real()), b_(cval.imag()) {};

      T real() const {return a_;}
      T imag() const {return b_;}

      wcomplex<T>& operator*=(const T& re) {a_ *= re; b_ *= re; return *this;}

      template<typename S>
      wcomplex<T>& operator*=(const std::complex<S>& z) {
        *this = (*this)*z;
        return *this;
      }

      template<typename S>
      wcomplex<T>& operator+=(const std::complex<S>& z) {
        *this = (*this)+z;
        return *this;
      }

      wcomplex<T>& operator+=(const wcomplex<T>& z) {
        *this = (*this)+z;
        return *this;
      }

      operator std::complex<double> () const {
        return std::complex<double>(a_.template convert_to<double>(), b_.template convert_to<double>());
      }

      template<typename S> S convert_to() const {
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
    operator+(const wcomplex<T>& z, const wcomplex<T>& w) {
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
    operator+(const wcomplex<T>& z, const std::complex<double>& w) {
      T a = z.real();
      T b = z.imag();
      double c = w.real();
      double d = w.imag();
      T x = a + c;
      T y = b + d;
      return wcomplex<T>(x, y);
    }

    template<class T>
    wcomplex<T>
    operator*(const wcomplex<T>& z, const wcomplex<T>& w)
    {
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
    operator*(const wcomplex<T>& z, const std::complex<S>& w) {
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
    operator*(const std::complex<S>& z, const wcomplex<T>& w) {
      return w*z;
    };

    template<typename T>
    wcomplex<T>
    operator*(const wcomplex<T>& z, double w) {
      T a = z.real();
      T b = z.imag();
      return wcomplex<T>(a * w, b * w);
    }

    template<typename T>
    wcomplex<T>
    operator*(double z, const wcomplex<T>& w) {
      return w*z;
    }

    template<class T>
    wcomplex<T>
    operator-(const wcomplex<T>& z, const wcomplex<T>& w) {
      T a = z.real();
      T b = z.imag();
      T c = w.real();
      T d = w.imag();
      T x = a - c;
      T y = b - d;
      return wcomplex<T>(x, y);
    }

    template<typename T>
    wcomplex<T>
    operator/(const wcomplex<T>& z, const wcomplex<T>& w)
    {
      T a = z.real();
      T b = z.imag();
      T c = w.real();
      T d = w.imag();
      T coeff = 1.0/(c*c + d*d);
      T x = (a * c + b * d)*coeff;
      T y = (-a * d + b * c)*coeff;
      return wcomplex<T>(x, y);
    }

    template<typename T>
    wcomplex<T>
    operator/(const wcomplex<T>& z, const std::complex<double>& w)
    {
      T a = z.real();
      T b = z.imag();
      double c = w.real();
      double d = w.imag();
      double coeff = 1.0/(c*c + d*d);
      T x = (a * c + b * d)*coeff;
      T y = (-a * d + b * c)*coeff;
      return wcomplex<T>(x, y);
    }

    template<typename T>
    bool
    operator==(const wcomplex<T>& z, const wcomplex<T>& w) {
      return (z.real()==w.real()) && (z.imag()==z.imag());
    }
  }
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const alps::cthyb::wcomplex<T>& val) {
  os << "( " << val.real() << " , " << val.imag() << " )";
  return os;
}

typedef boost::multiprecision::cpp_dec_float_50 EXTENDED_REAL;
typedef alps::cthyb::wcomplex<EXTENDED_REAL> EXTENDED_COMPLEX;

//Type traits
template<typename T>
struct ExtendedScalar {
  typedef EXTENDED_REAL value_type;
};

template<typename T>
struct ExtendedScalar<std::complex<T> > {
  typedef EXTENDED_COMPLEX value_type;
};

/**
 * Some auxially functions
 */
//template <unsigned Digits10, class ExponentType2_t, class Allocator = void>
//class cpp_dec_float
inline boost::multiprecision::cpp_dec_float_50 get_real(const boost::multiprecision::cpp_dec_float_50& x) {
  return x;
}

inline boost::multiprecision::cpp_dec_float_50 get_imag(const boost::multiprecision::cpp_dec_float_50& x) {
  return 0.0;
}

inline boost::multiprecision::cpp_dec_float_50 get_real(const std::complex<boost::multiprecision::cpp_dec_float_50>& x) {
  return x.real();
}

inline boost::multiprecision::cpp_dec_float_50 get_imag(const std::complex<boost::multiprecision::cpp_dec_float_50>& x) {
  return x.real();
}

inline boost::multiprecision::cpp_dec_float_50 myabs(boost::multiprecision::cpp_dec_float_50 x) {
  return boost::multiprecision::abs(x);
}

inline boost::multiprecision::cpp_dec_float_50
myabs(const alps::cthyb::wcomplex<boost::multiprecision::cpp_dec_float_50>& x) {
  return boost::multiprecision::sqrt(x.real()*x.real() + x.imag()*x.imag());
}

inline
bool my_isnan(boost::multiprecision::cpp_dec_float_50 x) {
  return boost::math::isnan(x);
}

inline
bool my_isnan(alps::cthyb::wcomplex<boost::multiprecision::cpp_dec_float_50> x) {
  return my_isnan(x.real()) || my_isnan(x.imag());
}

template<typename T2>
boost::multiprecision::cpp_dec_float_50 mypow(boost::multiprecision::cpp_dec_float_50 x, T2 N) {
  return boost::multiprecision::pow(x,N);
}

template<typename T2>
std::complex<boost::multiprecision::cpp_dec_float_50> mypow(std::complex<boost::multiprecision::cpp_dec_float_50> x, T2 N) {
  return boost::multiprecision::pow(x,N);
}
