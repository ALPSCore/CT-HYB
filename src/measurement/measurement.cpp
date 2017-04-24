#include "measurement.hpp"
#include "measurement.ipp"

typedef SlidingWindowManager<REAL_EIGEN_BASIS_MODEL> SW_REAL_MATRIX;
typedef SlidingWindowManager<COMPLEX_EIGEN_BASIS_MODEL> SW_COMPLEX_MATRIX;

void init_work_space(boost::multi_array<std::complex<double>, 3> &data, int num_flavors, int num_ir, int num_freq) {
  data.resize(boost::extents[num_flavors][num_flavors][num_ir]);
}

void init_work_space(boost::multi_array<std::complex<double>, 7> &data, int num_flavors, int num_ir, int num_freq) {
  data.resize(boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][num_ir][num_ir][num_freq]);
}

void
compute_w_tensor(
    int niw_positive,
    const FermionicIRBasis& basis_f,
    const BosonicIRBasis& basis_b,
    Eigen::Tensor<std::complex<double>,3>& w_tensor) {
  using dcomplex = std::complex<double>;

  const int dim_f = basis_f.dim();
  const int dim_b = basis_b.dim();

  std::vector<double> w(niw_positive);
  Eigen::Tensor<dcomplex,2> integral;
  for (int n=0; n < niw_positive; ++n) {
    w[n] = M_PI * (n+0.5);
  }

  std::vector<alps::gf::piecewise_polynomial<double>> prods(dim_f * dim_b);
  for (int lp = 0; lp < dim_f; ++lp) {
    for (int l = 0; l < dim_b; ++l) {
      prods[l + lp*dim_b] = alps::gf_extension::multiply(basis_b.basis_function(l), basis_f.basis_function(lp));
    }
  }
  alps::gf_extension::compute_integral_with_exp(w, prods, integral);

  w_tensor = Eigen::Tensor<dcomplex,3>(niw_positive, dim_b, dim_f);
  for (int lp = 0; lp < dim_f; ++lp) {
    for (int l = 0; l < dim_b; ++l) {
      for (int n = 0; n < niw_positive; ++n) {
        w_tensor(n, l, lp) = integral(n, l + lp*dim_b);
      }
    }
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
