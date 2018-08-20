#include "gf_basis.hpp"

double tau_to_x(double tau, double beta) {
    return  std::max(
            std::min(2*tau/beta - 1, 1.),
            -1.
    );
}

IRbasis::IRbasis(double Lambda, double beta, const std::string& file_name)
    : Lambda_(Lambda),
      beta_(beta),
      basis_f_(),
      basis_b_() {

    //auto file_name = "__INSTALL_PREFIX__" + std::string("/share/irbasis.h5");

    {
        std::ifstream ifs(file_name);
        if (!ifs.is_open()) {
            throw std::runtime_error(file_name + " does not exist!");
        }
    }

    //std::vector<double> Lambda_supported{10, 100, 1000, 10000};
    //if (std::find(Lambda_supported.begin(), Lambda_supported.end(), Lambda) == Lambda_supported.end()) {
        //throw std::runtime_error("Specified value of Lambda is not supported!");
    //}

    try {
        basis_f_ = irbasis::load("F", Lambda, file_name);
        basis_b_ = irbasis::load("B", Lambda, file_name);
    } catch (const std::exception& e) {
        std::cerr << "Error occured during reading a database file for IR basis at " + file_name + "!";
        throw std::runtime_error("Error occured during reading a database file for IR basis at " + file_name + "!");
    }

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
    auto trans = [&](double x){return std::sqrt(2/beta_) * x;};
    std::transform(val.begin(), val.end(), val.begin(), trans);
}

void
IRbasis::compute_Utau_B(double tau, std::vector<double> &val) const {
    assert(0 <= tau && tau <= beta_);
    basis_b_.ulx_all_l(tau_to_x(tau, beta_), val);
    auto trans = [&](double x){return std::sqrt(2/beta_) * x;};
    std::transform(val.begin(), val.end(), val.begin(), trans);
}
