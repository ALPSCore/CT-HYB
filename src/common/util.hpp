#pragma once

#include <boost/multi_array.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

#include <alps/mc/random01.hpp>

#include <iostream>
#include <math.h>
#include <vector>
#include <complex>

#include "wide_scalar.hpp"

template<typename T>
T mycast(std::complex<double> val);
template<typename T>
T myconj(T val);
template<typename T>
T mysign(T x);

double myabs(double x);

double myabs(std::complex<double> x);

double mymod(double x, double beta);

template<typename T>
bool my_isnan(T x);

double get_real(std::complex<double> x);

double get_imag(std::complex<double> x);

double get_real(double x);

double get_imag(double x);

template<typename T>
T mysign(T x) {
  return x / myabs(x);
}

template<typename T>
bool my_equal(T x, T y, double eps = 1E-8) {
  return std::abs(x - y) / std::max(std::abs(x), std::abs(y)) < eps;
}

template<typename T>
bool my_rdiff(T x, T y) {
  return std::abs(x - y) / std::max(std::abs(x), std::abs(y));
}

template<typename Derived>
inline int size1(const Eigen::EigenBase<Derived> &mat) {
  return mat.rows();
}

template<typename Derived>
inline int size2(const Eigen::EigenBase<Derived> &mat) {
  return mat.cols();
}

template<typename Derived1, typename Derived2, typename Derived3>
inline void matrix_right_multiply(Eigen::EigenBase<Derived1> &mat1,
                                  Eigen::EigenBase<Derived2> &mat2,
                                  Eigen::EigenBase<Derived3> &res) {
  res = mat1 * mat2;
}

template<typename T>
double maxAbsCoeff(const Eigen::SparseMatrix<T> &mat) {
  double maxval;
  int count = 0;
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<T>::InnerIterator it(mat, k); it; ++it) {
      maxval = count == 0 ? std::abs(it.value()) : std::max(std::abs(it.value()), maxval);
    }
  }
  return maxval;
}

inline double min_distance(double dist, double BETA);


template<typename SCALAR, typename M>
double spectral_norm_SVD(const M &mat) {
  typedef Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
  const double cutoff = 1E-15;
  const int rows = size1(mat);
  const int cols = size2(mat);
  if (rows == 0 || cols == 0) {
    return 0.0;
  }
  matrix_t mat_tmp(rows, cols);
  double max_abs = -1;
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      mat_tmp(i, j) = mat(i, j);
      max_abs = std::max(max_abs, std::abs(mat_tmp(i, j)));
    }
  }
  if (max_abs == 0.0) {
    return 0.0;
  }
  const double coeff = 1.0 / max_abs;
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      mat_tmp(i, j) *= coeff;
      if (std::abs(mat_tmp(i, j)) < cutoff) {
        mat_tmp(i, j) = 0.0;
      }
    }
  }
  //const double tmp = mat_tmp.squaredNorm();
  Eigen::JacobiSVD<matrix_t> svd(mat_tmp);
#ifndef NDEBUG
  const int size_SVD = svd.singularValues().size();
  for (int i = 0; i < size_SVD - 1; ++i) {
    assert(std::abs(svd.singularValues()[i]) >= std::abs(svd.singularValues()[i + 1]));
  }
  if (isnan(std::abs(svd.singularValues()[0]) / coeff)) {
    std::cout << "Norm is Nan" << std::endl;
    std::cout << "max_abs is " << max_abs << std::endl;
    std::cout << "coeff is " << coeff << std::endl;
    std::cout << "mat is " << std::endl << mat << std::endl;
    std::cout << "mat_tmp is " << std::endl << mat_tmp << std::endl;
    exit(-1);
  }
#endif
  const double norm = std::abs(svd.singularValues()[0]) / coeff;
  assert(!isnan(norm));
  return norm;
}

template<typename SCALAR, typename M>
double spectral_norm_diag(const M &mat) {
  typedef Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
  const double cutoff = 1E-15;
  const int rows = size1(mat);
  const int cols = size2(mat);
  if (rows == 0 || cols == 0) {
    return 0.0;
  }
  matrix_t mat_tmp(rows, cols);
  double max_abs = -1;
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      mat_tmp(i, j) = mat(i, j);
      max_abs = std::max(max_abs, std::abs(mat_tmp(i, j)));
    }
  }
  if (std::abs(max_abs) < 1E-300) {
    return 0.0;
  } else {
    const double coeff = 1.0 / max_abs;
    for (int j = 0; j < cols; ++j) {
      for (int i = 0; i < rows; ++i) {
        mat_tmp(i, j) *= coeff;
        if (std::abs(mat_tmp(i, j)) < cutoff) {
          mat_tmp(i, j) = 0.0;
        }
      }
    }
    if (mat_tmp.rows() > mat_tmp.cols()) {
      mat_tmp = mat_tmp.adjoint() * mat_tmp;
    } else {
      mat_tmp = mat_tmp * mat_tmp.adjoint();
    }
    const double max_abs2 = mat_tmp.cwiseAbs().maxCoeff();
    for (int j = 0; j < mat_tmp.cols(); ++j) {
      for (int i = 0; i < mat_tmp.rows(); ++i) {
        if (std::abs(mat_tmp(i, j)) < cutoff) {
          mat_tmp(i, j) = 0.0;
        }
      }
    }
    Eigen::SelfAdjointEigenSolver<matrix_t> esolv(mat_tmp, false);
    const double norm = std::sqrt(esolv.eigenvalues().cwiseAbs().maxCoeff()) / coeff;
    if (std::isnan(norm)) {
      std::cout << "Warning: spectral_norm_diag is NaN. max_abs = " << max_abs << " max_abs2 = " << max_abs2
                << std::endl;
      return 0.0;
    }
    //assert(!isnan(norm));
    return norm;
  }
}

//Extract real parts of boost::muliti_array
template<class SCALAR, int DIMENSION>
boost::multi_array<double, DIMENSION>
get_real_parts(const boost::multi_array<SCALAR, DIMENSION> &data) {
  boost::multi_array<double, DIMENSION> real_part(data.shape());
  std::transform(data.begin(), data.end(), real_part.begin(), get_real);
  return real_part;
};

//Extract imaginary parts of boost::muliti_array
template<class SCALAR, int DIMENSION>
boost::multi_array<double, DIMENSION>
get_imag_parts(const boost::multi_array<SCALAR, DIMENSION> &data) {
  boost::multi_array<double, DIMENSION> imag_part(data.shape());
  std::transform(data.begin(), data.end(), imag_part.begin(), get_imag);
  return imag_part;
}

template<class RNG>
double open_random(RNG &rng, double t1, double t2, double eps = 1e-10) {
  return (t1 - t2) * ((1 - eps) * rng() + eps / 2) + t2;
}




//Compute the determinant of a matrix avoiding underflow and overflow
//Note: This make a copy of the matrix.
/*
template<typename ReturnType, typename Derived>
ReturnType
safe_determinant(const Eigen::MatrixBase<Derived>& mat) {
    typedef typename Derived::RealScalar RealScalar;
    assert(mat.rows()==mat.cols());
    const int N = mat.rows();
    if (N==0) {
        return ReturnType(1.0);
    }
    Eigen::Matrix<typename Derived::Scalar,Eigen::Dynamic,Eigen::Dynamic> mat_copy(mat);
    const RealScalar max_coeff = mat_copy.cwiseAbs().maxCoeff();
    if (max_coeff==0.0) {
        return ReturnType(0.0);
    }
    mat_copy /= max_coeff;
    return ReturnType(mat_copy.determinant())*mypow(EXTENDED_REAL(max_coeff), N);
}

//Compute the inverse of a matrix avoiding underflow and overflow
//Note: This make a copy of the matrix.
template<typename Derived>
inline
void
safe_invert_in_place(Eigen::MatrixBase<Derived>& mat) {
    typedef typename Derived::RealScalar RealScalar;

    const int N = mat.rows();
    const RealScalar max_coeff = mat.cwiseAbs().maxCoeff();

    Eigen::Matrix<typename Derived::Scalar,Eigen::Dynamic,Eigen::Dynamic> mat_copy = mat/max_coeff;
    mat = mat_copy.inverse()/max_coeff;
}

template<class SCALAR>
inline SCALAR dsign(SCALAR s) {
    SCALAR abs_s = myabs(s);
    if (abs_s==SCALAR(0.0)) {
        throw std::runtime_error("dsign: s must not be zero");
    } else {
        return s/abs_s;
    }
}
 */

template<typename T>
struct AbsLessor {
  bool operator()(const T &t1, const T &t2) const {
    return (myabs(t1) < myabs(t2));
  }
};

template<typename T>
struct AbsGreater {
  bool operator()(const T &t1, const T &t2) const {
    return (myabs(t1) > myabs(t2));
  }
};

//for std::random_shuffle
class MyRandomNumberGenerator: public std::unary_function<unsigned int, unsigned int> {
 public:
  MyRandomNumberGenerator(alps::random01 &random) : random_(random) {};
  unsigned int operator()(unsigned int N) {
    return static_cast<unsigned int>(N * random_());
  }

 private:
  alps::random01 &random_;
};

/**
 * @brief pick n elements randombly from 0, 1, ..., N-1.
 * The results are given in a random order.
 */
template<class R>
std::vector<int> pickup_a_few_numbers(int N, int n, R &random01) {
  std::vector<int> list(N);
  for (int i = 0; i < N; ++i) {
    list[i] = i;
  }
  MyRandomNumberGenerator rnd(random01);
  std::random_shuffle(list.begin(), list.end(), rnd);
  list.resize(n);
  return list;
}

/*
 * Iterator over two sets
 */
template<class Set>
class TwoSetViewConstIterator;

template<class Set>
class TwoSetView {
 public:
  typedef TwoSetViewConstIterator<Set> const_iterator;
  TwoSetView(const Set &set1, const Set &set2) : set1_(set1), set2_(set2) {
    typename Set::const_iterator it_begin, it_end;
    if (set1_.empty() && set2_.empty()) {
      throw std::runtime_error("TwoSetView:proceed(): both sets are empty!");
    } else if (set1_.empty()) {
      it_begin = set2_.begin();
      it_end = set2_.end();
    } else if (set2_.empty()) {
      it_begin = set1_.begin();
      it_end = set1_.end();
    } else {
      if (*set1.begin() < *set2.begin()) {
        it_begin = set1.begin();
      } else if (*set1.begin() > *set2.begin()) {
        it_begin = set2.begin();
      } else {
        throw std::runtime_error("TwoSetView: found duplicate elements!");
      }
      if (*set1.rbegin() < *set2.rbegin()) {
        it_end = set2.end();
      } else if (*set1.rbegin() > *set2.rbegin()) {
        it_end = set1.end();
      } else {
        throw std::runtime_error("TwoSetView: found duplicate elements!");
      }
    }

    typename Set::const_iterator it1_next(set1.begin()), it2_next(set2.begin());
    if (it_begin == set1.begin()) {
      ++ it1_next;
    }
    if (it_begin == set2.begin()) {
      ++ it2_next;
    }
    const_it_begin_ = const_iterator(0, it_begin, it1_next, it2_next, set1.end(), set2.end());
    const_it_end_ = const_iterator(set1.size() + set2.size(), it_end, set1.end(), set2.end(), set1.end(), set2.end());
  };

  const const_iterator& begin() const {
    return const_it_begin_;
  }

  const const_iterator& end() const {
    return const_it_end_;
  }

 private:
  const Set &set1_, set2_;
  const_iterator const_it_begin_, const_it_end_;
};

template<class Set>
class TwoSetViewConstIterator {
  friend class TwoSetView<Set>;
 public:
  typedef typename Set::value_type value_type;
  TwoSetViewConstIterator<Set> &operator++() {
    this->proceed();
    return *this;
  }

  TwoSetViewConstIterator<Set> operator++(int num) {
    TwoSetViewConstIterator<Set> copy(*this);
    copy.proceed();
    return copy;
  }

  bool operator==(const TwoSetViewConstIterator<Set> &it_r) const {
    return this->index_ == it_r.index_;
  }

  bool operator!=(const TwoSetViewConstIterator<Set> &it_r) const {
    return !(*this == it_r);
  }

  const value_type &operator*() const {
    return *it_;
  }

  value_type operator*() {
    return *it_;
  }

  const value_type &operator->() const {
    return it_.operator->();
  }

  value_type &operator->() {
    return it_.operator->();
  }

 private:
  TwoSetViewConstIterator() : index_(0) {}

  TwoSetViewConstIterator(
      int index,
      typename Set::const_iterator it,
      typename Set::const_iterator it1_next,
      typename Set::const_iterator it2_next,
      typename Set::const_iterator it1_end,
      typename Set::const_iterator it2_end
  )
      : index_(index),
        it_(it),
        it1_next_(it1_next),
        it2_next_(it2_next),
        it1_end_(it1_end),
        it2_end_(it2_end) {}

  void proceed() {
    if (it1_next_ == it1_end_ && it2_next_ == it2_end_) {
      //do nothing
    } else if (it1_next_ == it1_end_) {
      it_ = it2_next_;
      ++it2_next_;
    } else if (it2_next_ == it2_end_) {
      it_ = it1_next_;
      ++ it1_next_;
    } else if (*it1_next_ < *it2_next_) {
      it_ = it1_next_;
      ++ it1_next_;
    } else if (*it1_next_ > *it2_next_) {
      it_ = it2_next_;
      ++ it2_next_;
    } else {
      throw std::runtime_error("TwoSetViewConstIterator:proceed(): found duplicate elements!");
    }

    ++index_;
    std::cout << "index " << index_ << std::endl;
  }

  int index_;
  typename Set::const_iterator it_, it1_next_, it2_next_;
  typename Set::const_iterator it1_end_, it2_end_;
};


template<typename T>
struct PruneHelper {
  PruneHelper(double eps) : eps_(eps) { };
  bool operator()(const int &row, const int &col, const T &value) const {
    return (std::abs(value) > eps_);
  }
  const double eps_;
};

inline void check_true(bool b, const std::string& str="") {
  if (!b) {
    throw std::runtime_error("Something got wrong! " + str);
  }
}