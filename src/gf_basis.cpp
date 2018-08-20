#include "gf_basis.hpp"

double tau_to_x(double tau, double beta) {
    return  std::max(
            std::min(2*tau/beta - 1, 1.),
            -1.
    );
}

std::vector<double> linspace(double minval, double maxval, int N, bool include_last_point = true) {
    int end = include_last_point ? N : N-1;
    std::vector<double> r(end);
    for (int i = 0; i < end; ++i) {
        r[i] = i * (maxval - minval) / (N - 1.0) + minval;
    }
    return r;
}

std::vector<double> find_zeros(
        const std::function<double(double)>& f,
        double delta = 1e-12
) {
    int N = 10000;
    double de_cutoff = 3.0;

    std::vector<double> tx_vec = linspace(-de_cutoff, de_cutoff, N);
    std::vector<double> x_vec(N), zeros;
    for (int i = 0; i < N; ++i) {
        x_vec[i] = tanh(0.5 * M_PI * sinh(tx_vec[i]));
    }

    for (int i = 0; i < N-1; ++i) {
        if (f(x_vec[i]) * f(x_vec[i+1]) < 0) {
            double x_left = x_vec[i];
            double fx_left = f(x_vec[i]);

            double x_right = x_vec[i+1];
            double fx_right = f(x_vec[i+1]);

            while (x_right-x_left > delta) {
                double x_mid = (x_left+x_right)/2;
                if (fx_left * f(x_mid) > 0) {
                    x_left = x_mid;
                } else {
                    x_right = x_mid;
                }
            }
            zeros.push_back((x_left+x_right)/2);
        }
    }

    return zeros;
};

IRbasis::IRbasis(double Lambda, double beta, const std::string& file_name)
    : Lambda_(Lambda),
      beta_(beta),
      basis_f_(),
      basis_b_(),
      bin_edges_()
{

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

    auto zeros_x_f = find_zeros(
            [&](double x){return basis_f_.ulx(basis_f_.dim()-1, x);}
    );
    bin_edges_.push_back(0);
    std::transform(
            zeros_x_f.begin(), zeros_x_f.end(), std::back_inserter(bin_edges_),
            [&](double x) {return beta_ * 0.5 * (x+1);}
            );
    bin_edges_.push_back(beta_);
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
