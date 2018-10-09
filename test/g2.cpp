#include <boost/random.hpp>
#include "../src/measurement/measurement.hpp"
#include "../src/measurement/measurement.ipp"

#include <gtest.h>

TEST(G2, MeasureByHyb) {
  boost::random::mt19937 gen(100);
  boost::uniform_real<> uni_dist(0, 1);

  using SCALAR = double;

  int num_flavors = 4;

  double beta = 1.0;

  int num_freq_f = 10;
  int num_freq_b = 10;

  // List of fermion frequencies
  std::vector<int> freq_index_f(num_freq_f);
  for (int i=0; i<num_freq_f; ++i) {
    freq_index_f[i] = i - num_freq_f / 2;
  }

  // List of boson frequencies
  std::vector<int> freq_index_b(num_freq_b);
  for (int i=0; i<num_freq_b; ++i) {
    freq_index_b[i] = i;
  }

  for (auto k : std::vector<int>{4}) {
    std::vector<psi> creation_ops;
    std::vector<psi> annihilation_ops;
    Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic> M(k, k);

    for (int i=0; i<k; ++i) {
      creation_ops.push_back(psi(beta*uni_dist(gen), CREATION_OP, static_cast<int>(num_flavors*uni_dist(gen))));
      annihilation_ops.push_back(psi(beta*uni_dist(gen), ANNIHILATION_OP, static_cast<int>(num_flavors*uni_dist(gen))));
    }

    for (int j=0; j<k; ++j) {
      for (int i=0; i<k; ++i) {
        M(i,j) = uni_dist(gen);
      }
    }

    auto extents = boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][num_freq_f][num_freq_f][num_freq_b];

    boost::multi_array<std::complex<double>,7> result_k4(extents);
    std::fill(result_k4.origin(), result_k4.origin() + result_k4.num_elements(), 0.0);
    measure_G2_k4_PH<SCALAR>(
      beta,
      num_flavors,
      k,
      1.0,
      M,
      creation_ops,
      annihilation_ops,
      freq_index_f,
      freq_index_b,
      result_k4);

    boost::multi_array<std::complex<double>,7> result_k2(extents);
    std::fill(result_k2.origin(), result_k2.origin() + result_k2.num_elements(), 0.0);
    measure_G2_k2_PH<SCALAR>(
        beta,
        num_flavors,
        k,
        1.0,
        M,
        creation_ops,
        annihilation_ops,
        freq_index_f,
        freq_index_b,
        result_k2);

    auto it_k2 = result_k2.origin();
    for (auto it_k4 = result_k4.origin(); it_k4 != result_k4.origin() + result_k4.num_elements(); ++it_k4) {
      auto diff = std::abs(*it_k2 - *it_k4);
      ASSERT_TRUE(diff < 1E-10);
      ++it_k2;
    }

    std::vector<std::tuple <int,int,int> > freqs;
    for (int i=0; i<num_freq_f; i+=3) {
      for (int j=0; j<num_freq_f; j+=3) {
        for (int k=0; k<num_freq_b; k+=3) {
          freqs.push_back(std::make_tuple(freq_index_f[i], freq_index_f[j], freq_index_b[k]));
        }
      }
    }
    auto extents_general = boost::extents[num_flavors][num_flavors][num_flavors][num_flavors][freqs.size()];
    boost::multi_array<std::complex<double>,5> result_k2_general(extents_general);
    std::fill(result_k2_general.origin(), result_k2_general.origin() + result_k2_general.num_elements(), 0.0);
    measure_G2_k2_PH_impl(beta, num_flavors, k, 1.0, M,
                          creation_ops,
                          annihilation_ops,
                          freqs,
                          result_k2_general);
    int index = 0;
    for (int i=0; i<num_freq_f; i+=3) {
      for (int j = 0; j < num_freq_f; j += 3) {
        for (int k = 0; k < num_freq_b; k += 3) {
          auto diff = std::abs(
              result_k2[0][0][0][0][i][j][k] - result_k2_general[0][0][0][0][index]
          );
          //std::cout << freq_index_f[i] << " " << freq_index_f[j] << " " << freq_index_b[k] << result_k4[0][0][0][0][i][j][k] << " " << result_k2_general[0][0][0][0][index] << std::endl;
          ASSERT_TRUE(diff < 1E-10);
          ++ index;
        }
      }
    }


    //Eigen::Tensor<SCALAR,7> r = measure_g2(beta, num_flavors, p_basis_f, p_basis_b, creation_ops, annihilation_ops, M);
    //Eigen::Tensor<SCALAR,7> r_ref = measure_g2_ref(beta, num_flavors, p_basis_f, p_basis_b, creation_ops, annihilation_ops, M);
    //Eigen::Tensor<SCALAR,7> rdiff = (r - r_ref).abs();
    //Eigen::Tensor<SCALAR,0> diff = rdiff.sum();
    //ASSERT_TRUE(diff.coeff() < 1E-10);
  }

}
