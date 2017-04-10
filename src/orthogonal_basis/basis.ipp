#include "basis.hpp"

template<class Base>
const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
IrBasis<Base>::Tnl(int n_iw) const {
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

  if (Tnl_.rows() != n_iw) {
    Tnl_.resize(n_iw, dim());
    boost::multi_array<std::complex<double>,2> tmp;
    p_basis_->compute_Tnl(0, n_iw-1, tmp);
    for (int n = 0; n < n_iw; ++n) {
      for (int l = 0; l < dim(); ++l) {
        Tnl_(n,l) = tmp[n][l];
      }
    }
  }

  return Tnl_;
}