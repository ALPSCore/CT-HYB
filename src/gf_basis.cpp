#include "gf_basis.hpp"

double tau_to_x(double tau, double beta) {
    return  std::max(
            std::min(2*tau/beta - 1, 1.),
            -1.
    );
}

IRbasis::IRbasis(double Lambda, double beta)
    : Lambda_(Lambda),
      beta_(beta),
      basis_f_(irbasis::load("F", Lambda, "__INSTALL_PREFIX__"+std::string("/share/irbasis.h5"))),
      basis_b_(irbasis::load("B", Lambda, "__INSTALL_PREFIX__"+std::string("/share/irbasis.h5"))) {

};

Eigen::MatrixXcd
IRbasis::compute_Unl_F(int niw) const {
    std::vector<long long> nvec(niw);
    for (int i=0; i<niw; ++i) {
        nvec[i] = i;
    }
    return std::sqrt(2/beta_) * convert_to_eigen_matrix(basis_f_.compute_unl(nvec));
}

Eigen::MatrixXcd
IRbasis::compute_Unl_B(int niw) const {
    std::vector<long long> nvec(niw);
    for (int i=0; i<niw; ++i) {
        nvec[i] = i;
    }
    return std::sqrt(2/beta_) * convert_to_eigen_matrix(basis_b_.compute_unl(nvec));
}

void
IRbasis::compute_Utau_F(double tau, std::vector<double> &val) const {
    assert(0 <= tau && tau <= beta_);
    basis_f_.ulx_all_l(tau_to_x(tau, beta_), val);
}

void
IRbasis::compute_Utau_B(double tau, std::vector<double> &val) const {
    assert(0 <= tau && tau <= beta_);
    basis_b_.ulx_all_l(tau_to_x(tau, beta_), val);
}
