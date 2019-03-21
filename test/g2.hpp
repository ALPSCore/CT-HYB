#include <vector>
#include <boost/multi_array.hpp>

/**
 * Measure only straight diagrams
 * @tparam SCALAR
 * @param beta
 * @param num_flavors
 * @param num_phys_rows
 * @param overall_coeff
 * @param M_prime
 * @param creation_ops
 * @param annihilation_ops
 * @param freq_index_f
 * @param freq_index_b
 * @param result
 */
template<typename SCALAR>
void measure_G2_k4_PH(
    double beta,
    int num_flavors,
    int num_phys_rows,
    SCALAR overall_coeff,
    const Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic>& M_prime,
    const std::vector<psi>& creation_ops,
    const std::vector<psi>& annihilation_ops,
    const std::vector<int>& freq_index_f,
    const std::vector<int>& freq_index_b,
    boost::multi_array<std::complex<double>, 7> &result
) {
  // List of fermion frequencies
  auto iwn_f_min = freq_index_f[0];
  double min_freq_f = M_PI * (2 * iwn_f_min + 1) / beta;
  auto num_freq_f = freq_index_f.size();
  auto num_freq_b = freq_index_b.size();

  boost::multi_array<std::complex<double>, 3>
      exp_f(boost::extents[num_phys_rows][num_phys_rows][num_freq_f]);//annihilator, creator, freq_f
  boost::multi_array<std::complex<double>, 3>
      exp_b(boost::extents[num_phys_rows][num_phys_rows][num_freq_b]);//annihilator, creator, freq_b
  double tau_diff, sign_mod;
  for (int k = 0; k < num_phys_rows; k++) {
    for (int l = 0; l < num_phys_rows; l++) {
      double argument = annihilation_ops[k].time() - creation_ops[l].time();

      std::tie(tau_diff, sign_mod) = mod_beta(argument, beta);

      for (int freq = 0; freq < num_freq_f; ++freq) {
        auto wn =  M_PI * (2 * freq_index_f[freq] + 1) / beta;
        exp_f[k][l][freq] = sign_mod * std::exp(std::complex<double>(0, wn * tau_diff));
      }

      for (int freq = 0; freq < num_freq_b; ++freq) {
        auto wn =  M_PI * (2 * freq_index_b[freq]) / beta;
        exp_b[k][l][freq] = std::exp(std::complex<double>(0, wn * tau_diff));
      }
    }
  }

  auto extents = boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][num_freq_f][num_freq_f][num_freq_b];
  boost::multi_array<std::complex<double>, 7> result_tmp(extents);
  std::fill(result_tmp.origin(), result_tmp.origin() + result_tmp.num_elements(), 0.0);
  for (int a = 0; a < num_phys_rows; ++a) {
    for (int b = 0; b < num_phys_rows; ++b) {
      for (int c = 0; c < num_phys_rows; ++c) {
        for (int d = 0; d < num_phys_rows; ++d) {
          /*
           * Delta convention
           * M_ab  M_ad  M_a*
           * M_cb  M_cd  M_c*
           * M_*b  M_*d  M_**
           *
           * Here, M is in F convention. Rows and columns must be swapped.
           */
          SCALAR det = M_prime(b,a) * M_prime(d,c);

          if (det == 0.0) {
            continue;
          }

          auto fa = annihilation_ops[a].flavor();
          auto fb = creation_ops[b].flavor();
          auto fc = annihilation_ops[c].flavor();
          auto fd = creation_ops[d].flavor();
          for (int freq_f1 = 0; freq_f1 < num_freq_f; ++freq_f1) {
            for (int freq_f2 = 0; freq_f2 < num_freq_f; ++freq_f2) {
              for (int freq_b = 0; freq_b < num_freq_b; ++freq_b) {
                result_tmp[fa][fb][fc][fd][freq_f1][freq_f2][freq_b] +=
                    det * exp_f[a][b][freq_f1] * exp_f[c][d][freq_f2] * exp_b[a][d][freq_b];
              }
            }
          }
        }
      }
    }
  }

  auto it2 = result.origin();
  for (auto it = result_tmp.origin(); it != result_tmp.origin() + result_tmp.num_elements(); ++it) {
    *it2 += (*it) * overall_coeff;
    ++it2;
  }
}

/**
 * Does the same job as measure_G2_k4_PH. But computational cost scales as O(k^2) where k is the matrix size for reconnection
 * Only straight diagrams (denoted as [H}artree) is measured.
 * @tparam SCALAR
 * @param beta
 * @param num_flavors
 * @param num_phys_rows
 * @param overall_coeff
 * @param M_prime
 * @param creation_ops
 * @param annihilation_ops
 * @param freq_index_f
 * @param freq_index_b
 * @param result
 */
template<typename SCALAR>
void measure_G2_k2_PH(
    double beta,
    int num_flavors,
    int num_phys_rows,
    SCALAR overall_coeff,
    const Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic>& M_prime,
    const std::vector<psi>& creation_ops,
    const std::vector<psi>& annihilation_ops,
    const std::vector<int>& freq_index_f,
    const std::vector<int>& freq_index_b,
    boost::multi_array<std::complex<double>, 7> &result
) {
  using dcomplex = std::complex<double>;

  // List of fermion frequencies
  auto num_freq_f = freq_index_f.size();
  auto num_freq_b = freq_index_b.size();

  std::vector<matsubara_freq_point_PH> freqs(num_freq_f * num_freq_f * num_freq_b);

  auto idx = 0;
  for (auto idx_f1 = 0; idx_f1<num_freq_f; ++idx_f1) {
    for (auto idx_f2 = 0; idx_f2<num_freq_f; ++idx_f2) {
      for (auto idx_b = 0; idx_b<num_freq_b; ++idx_b) {
        freqs[idx] = matsubara_freq_point_PH(freq_index_f[idx_f1], freq_index_f[idx_f2], freq_index_b[idx_b]);
        ++ idx;
      }
    }
  }

  std::vector<std::pair<int,int>> two_freqs_vec;
  std::unordered_map<std::pair<int,int>, int, HashIntPair> two_freqs_map;
  make_two_freqs_list(freqs, two_freqs_vec, two_freqs_map);

  auto extents_out = boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][freqs.size()];
  boost::multi_array_ref<dcomplex,5> out_buffer(result.origin(), extents_out);
  measure_G2_k2_PH_impl(beta, num_flavors, num_phys_rows, overall_coeff, M_prime,
                        creation_ops, annihilation_ops, freqs, two_freqs_vec, two_freqs_map, out_buffer);
}
