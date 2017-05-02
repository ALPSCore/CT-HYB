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

void
compute_tensor_from_H_to_F_term(
    int niw_positive,
    const FermionicIRBasis& basis_f,
    const BosonicIRBasis& basis_b,
    Eigen::Tensor<std::complex<double>,6>& results) {
  using dcomplex = std::complex<double>;

  const int dim_f = basis_f.dim();
  const int dim_b = basis_b.dim();

  Eigen::Tensor<std::complex<double>,3> w_tensor;
  compute_w_tensor(niw_positive, basis_f, basis_b, w_tensor);

  auto Tnl_f = basis_f.Tnl(niw_positive);
  auto Tnl_b = basis_b.Tnl(niw_positive);

  results = Eigen::Tensor<dcomplex,6>(dim_f, dim_f, dim_b, dim_f, dim_f, dim_b);
  results.setZero();
  for (int lp3 = 0; lp3 < dim_b; ++lp3) {
    std::cout << "lp3 " << lp3 << std::endl;
    for (int lp2 = 0; lp2 < dim_f; ++lp2) {
      for (int lp1 = 0; lp1 < dim_f; ++lp1) {
        for (int l3 = 0; l3 < dim_b; ++l3) {
          for (int l2 = 0; l2 < dim_f; ++l2) {
            for (int l1 = 0; l1 < dim_f; ++l1) {
              dcomplex tmp = 0.0;
              //Note: the summation is taken only for positive Matsubara frequencies since
              // T_{-n,l}^* = T_{n,l}, w_tensor(-n,l,l') = w_tensor(n,l,l')^*.
              for (int n = 0; n < niw_positive; ++n) {
                auto z = w_tensor(n,l3,lp1) * std::conj(w_tensor(n,lp3,l1) * Tnl_f(n,l2)) * Tnl_f(n,lp2);
                tmp += z;
              }
              results(l1, l2, l3, lp1, lp2, lp3) = - 2*tmp.real() * ((l2+lp2)%2 == 0 ? 1.0 : -1.0);
            }
          }
        }
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
