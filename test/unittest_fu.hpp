#include <complex>
#include <algorithm>
#include <limits>
#include <functional>

#include <boost/random.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/random.hpp>
#include <boost/multi_array.hpp>
#include <boost/range/irange.hpp>

//To avoid compiler errors for intel compiler on a cray machine
#define GTEST_USE_OWN_TR1_TUPLE 1

#include <gtest.h>

#include <alps/fastupdate/determinant_matrix.hpp>
#include <alps/fastupdate/determinant_matrix_partitioned.hpp>
#include <alps/fastupdate/fastupdate_formula.hpp>
#include "util_fu.hpp"

template<typename Derived>
inline int num_cols(const Eigen::MatrixBase<Derived> &m) {
  return m.cols();
}

template<typename Derived>
inline int num_rows(const Eigen::MatrixBase<Derived> &m) {
  return m.rows();
}


template<class M>
void randomize_matrix(M& mat, size_t seed=100) {
  boost::random::mt19937 gen;
  boost::random::uniform_01<double> dist;
  gen.seed(seed);

  for (int j=0; j<num_cols(mat); ++j) {
    for (int i=0; i<num_rows(mat); ++i) {
      mat(i,j) = dist(gen);
    }
  }
}

//for std::random_shuffle
class rs_shuffle : public std::unary_function<unsigned int, int unsigned> {
public:
  rs_shuffle(boost::mt19937 &gen) : gen_(gen) {};
  unsigned int operator()(unsigned int N) {
    boost::uniform_int<> dist(0,N-1);
    return dist(gen_);
  }

private:
  boost::random::mt19937& gen_;
};

/*creation operator class*/
class c_or_cdagger
{
public:
  typedef double itime_type;
  c_or_cdagger() : flavor_(-1), t_(-1.0) {};
  c_or_cdagger(int flavor, double t)
  {
    flavor_ = flavor;
    t_ = t;
  }
  virtual ~c_or_cdagger() {}

  double time() const {return t_;}
  int flavor() const {return flavor_;}

private:
  int flavor_;
  double t_;
};

inline double operator_time(const c_or_cdagger &op) {
  return op.time();
}

inline int operator_flavor(const c_or_cdagger &op) {
  return op.flavor();
}

inline bool operator<(const c_or_cdagger& op1, const c_or_cdagger& op2) {
  return operator_time(op1) < operator_time(op2);
}

inline bool operator==(const c_or_cdagger& op1, const c_or_cdagger& op2) {
  return operator_time(op1) == operator_time(op2);
}

class creator : public c_or_cdagger {
public:
  creator() : c_or_cdagger() {};
  creator(int flavor, double t) : c_or_cdagger(flavor, t) {};
};

class annihilator : public c_or_cdagger {
public:
  annihilator() : c_or_cdagger() {};
  annihilator(int flavor, double t) : c_or_cdagger(flavor, t) {};
};

//Interpolation of G0
template<typename T>
struct OffDiagonalG0 {
  OffDiagonalG0 (double beta, int n_flavor, const std::vector<double>& E, const boost::multi_array<T,2>& phase) : beta_(beta), n_flavor_(n_flavor), E_(E), phase_(phase) {}

  int nflavor() const {return n_flavor_;}
  int num_flavors() const {return n_flavor_;}
  bool is_connected(int flavor, int flavor2) const {
    //return flavor%2==flavor2%2;
    return flavor==flavor2;
  }

  T operator() (const annihilator& c_op, const creator& cdagg_op) const {
    if (!is_connected(operator_flavor(c_op), operator_flavor(cdagg_op))) {
      return 0.0;
    }
    const double dt = c_op.time()-cdagg_op.time();
    double dt_tmp = dt;
    if (dt_tmp > beta_) dt_tmp -= beta_;
    if (dt_tmp < 0) dt_tmp += beta_;

    if (c_op.flavor()==cdagg_op.flavor()) {
      const double value_at_half_beta = E_[c_op.flavor()];
      const double a = (2-4*value_at_half_beta)/(beta_*beta_);
      return 2*(value_at_half_beta + a*std::pow(dt_tmp-0.5*beta_, 2));
    } else {
      return (-dt+0.5*beta_)/(2*beta_)*phase_[c_op.flavor()][cdagg_op.flavor()];
    }
  }

  double beta_;
  int n_flavor_;
  std::vector<double> E_;
  boost::multi_array<T,2> phase_;
};



typedef ::testing::Types<
  alps::fastupdate::DeterminantMatrix<
    std::complex<double>,
    OffDiagonalG0<std::complex<double> >,
    creator,
    annihilator>,
  alps::fastupdate::DeterminantMatrixPartitioned<
    std::complex<double>,
    OffDiagonalG0<std::complex<double> >,
    creator,
    annihilator
  >
>TestTypes;
