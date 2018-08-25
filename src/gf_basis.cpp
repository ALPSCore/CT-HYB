#include "gf_basis.hpp"

double tau_to_x(double tau, double beta) {
  return std::max(
      std::min(2 * tau / beta - 1, 1.),
      -1.
  );
}

std::vector<double> linspace(double minval, double maxval, int N, bool include_last_point = true) {
  int end = include_last_point ? N : N - 1;
  std::vector<double> r(end);
  for (int i = 0; i < end; ++i) {
    r[i] = i * (maxval - minval) / (N - 1.0) + minval;
  }
  return r;
}

std::vector<double> find_zeros(
    const std::function<double(double)> &f,
    double delta = 1e-12
) {
  int N = 10000;
  double de_cutoff = 3.0;

  std::vector<double> tx_vec = linspace(-de_cutoff, de_cutoff, N);
  std::vector<double> x_vec(N), zeros;
  for (int i = 0; i < N; ++i) {
    x_vec[i] = tanh(0.5 * M_PI * sinh(tx_vec[i]));
  }

  for (int i = 0; i < N - 1; ++i) {
    if (f(x_vec[i]) * f(x_vec[i + 1]) < 0) {
      double x_left = x_vec[i];
      double fx_left = f(x_vec[i]);

      double x_right = x_vec[i + 1];
      double fx_right = f(x_vec[i + 1]);

      while (x_right - x_left > delta) {
        double x_mid = (x_left + x_right) / 2;
        if (fx_left * f(x_mid) > 0) {
          x_left = x_mid;
        } else {
          x_right = x_mid;
        }
      }
      zeros.push_back((x_left + x_right) / 2);
    }
  }

  return zeros;
};

IRbasis::IRbasis(const alps::params &params)
    : Lambda_(params["measurement.Lambda"]),
      beta_(params["model.beta"]),
      basis_f_(),
      basis_b_(),
      bin_edges_() {

  //auto file_name = "__INSTALL_PREFIX__" + std::string("/share/irbasis.h5");

  std::string file_name(params["measurement.IRbasis_database_file"].as<std::string>());
  {
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
      throw std::runtime_error(file_name + " does not exist!");
    }
  }

  try {
    basis_f_ = irbasis::load("F", Lambda_, file_name);
    basis_b_ = irbasis::load("B", Lambda_, file_name);
  } catch (const std::exception &e) {
    std::cerr << "Error occured during reading a database file for IR basis at " + file_name + "!";
    throw std::runtime_error("Error occured during reading a database file for IR basis at " + file_name + "!");
  }

  auto count_basis = [&](const irbasis::basis &basis, double cutoff) {
      int dim = 0;
      for (int l = 0; l < basis.dim(); ++l) {
        if (basis.sl(l) / basis.sl(0) > cutoff)
          dim += 1;
      }
      return dim;
  };
  dim_F_ = count_basis(basis_f_, 1e-2);
  dim_B_ = count_basis(basis_b_, 1e-2);

  auto zeros_x_f = find_zeros(
      [&](double x) { return basis_f_.ulx(dim_F_ - 1, x); }
  );
  bin_edges_.push_back(0);
  std::transform(
      zeros_x_f.begin(), zeros_x_f.end(), std::back_inserter(bin_edges_),
      [&](double x) { return beta_ * 0.5 * (x + 1); }
  );
  bin_edges_.push_back(beta_);

  // Load bins for measuring G4pt
  {
    alps::hdf5::archive f_4pt(params["measurement.G2.IRbasis_4pt_database_file"], "r");
    std::stringstream ss;
    ss << "/Lambda";
    ss << std::fixed << std::setprecision(1) << Lambda_;
    ss << "-dim";
    ss << std::fixed << std::setprecision(0) << dim_F_;
    f_4pt[ss.str() + "/num_bins"] >> num_bins_4pt_;

    bin_volume_4pt_.resize(num_bins_4pt_);
    bin_index_4pt_.resize(num_bins_4pt_);
    bin_centroid_4pt_.resize(num_bins_4pt_);

    std::vector<int> index_tmp(6);
    norm_const_4pt_ = 0;
    auto vol_ratio = std::pow(beta_/2, 3);//x -> tau
    std::vector<double> tmp_array;
    alps::numerics::tensor<double,2> t;
    for (int ib = 0; ib < num_bins_4pt_; ++ib) {
      auto path = ss.str() + "/bin" + std::to_string(ib);
      f_4pt[path + "/volume"] >> bin_volume_4pt_[ib];
      bin_volume_4pt_[ib] *= vol_ratio;
      f_4pt[path + "/index"] >> index_tmp;
      for (int i = 0; i < 6; ++i) {
        bin_index_4pt_[ib][i] = index_tmp[i];
      }
      norm_const_4pt_ += 1 / bin_volume_4pt_[ib];
      bin_index_map_4pt_[bin_index_4pt_[ib]] = ib;
      f_4pt[path + "/coords"] >> t;

      std::fill(bin_centroid_4pt_[ib].begin(), bin_centroid_4pt_[ib].end(), 0.0);
      auto nv = t.shape()[0];
      for (int v=0; v<nv; ++v) {
        for (int i=0; i<3; ++i) {
          bin_centroid_4pt_[ib][i] += 0.5*beta_*(t(v, i)+1)/nv;
        }
      }
    }
  }

  check();
};

void
IRbasis::check() const {
  // Check bins for 4pt Green's function
  double vol_sum = 0.0;
  for (int ib=0; ib < num_bins_4pt(); ++ib) {
    auto centroid = bin_centroid_4pt(ib);
    auto ib_centroid = get_bin_index(centroid[0], centroid[1], centroid[2], 0);
    vol_sum += bin_volume_4pt(ib);
    if (ib != ib_centroid) {
      throw std::runtime_error("Something went wrong with bins for 4pt Green's function!");
    }
  }

  if (std::abs(vol_sum - std::pow(beta(), 3)) > 1e-5) {
    throw std::runtime_error("Something went wrong with bins for 4pt Green's function!");
  }
}

Eigen::MatrixXcd
IRbasis::compute_Unl_F(int niw) const {
  std::vector<long long> nvec(niw);
  for (int i = 0; i < niw; ++i) {
    nvec[i] = i;
  }
  return std::sqrt(2 / beta_) * convert_to_eigen_matrix(basis_f_.compute_unl(nvec));
}

Eigen::MatrixXcd
IRbasis::compute_Unl_B(int niw) const {
  std::vector<long long> nvec(niw);
  for (int i = 0; i < niw; ++i) {
    nvec[i] = i;
  }
  return std::sqrt(2 / beta_) * convert_to_eigen_matrix(basis_b_.compute_unl(nvec));
}

void
IRbasis::compute_Utau_F(double tau, std::vector<double> &val) const {
  assert(0 <= tau && tau <= beta_);
  basis_f_.ulx_all_l(tau_to_x(tau, beta_), val);
  auto trans = [&](double x) { return std::sqrt(2 / beta_) * x; };
  std::transform(val.begin(), val.end(), val.begin(), trans);
}

void
IRbasis::compute_Utau_B(double tau, std::vector<double> &val) const {
  assert(0 <= tau && tau <= beta_);
  basis_b_.ulx_all_l(tau_to_x(tau, beta_), val);
  auto trans = [&](double x) { return std::sqrt(2 / beta_) * x; };
  std::transform(val.begin(), val.end(), val.begin(), trans);
}
