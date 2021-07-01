#include "unittest_solver.hpp"

TEST(Clustering, 2D_percolation) {
  const int L = 50;
  const int N = L * L;
  const double p = 0.2;

  Clustering cl(N);
  boost::multi_array<bool, 2> connected(boost::extents[N][N]);
  std::fill(connected.origin(), connected.origin() + connected.num_elements(), false);

  boost::random::mt19937 gen(100);
  boost::uniform_real<> uni_dist(0, 1);

  for (int x = 0; x < L; ++x) {
    for (int y = 0; y < L; ++y) {
      for (int xx = -1; xx < 2; ++xx) {
        for (int yy = -1; yy < 2; ++yy) {
          if (xx * xx + yy * yy != 1) {
            continue;
          }
          if (x + xx < 0 || x + xx >= L) {
            continue;
          }
          if (y + yy < 0 || y + yy >= L) {
            continue;
          }
          const int src = x * L + y;
          const int dst = (x + xx) * L + (y + yy);
          if (uni_dist(gen) < p) {
            connected[src][dst] = true;
            connected[dst][src] = true;
          }
        }
      }
    }
  }

  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      if (connected[i][j]) {
        cl.connect_vertices(i, j);
      }
    }
  }

  cl.finalize_labeling();

  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      if (connected[i][j]) {
        ASSERT_TRUE(cl.get_cluster_label(i) == cl.get_cluster_label(j));
      }
    }
  }
  //  std::cout << "# of clusters " << cl.get_num_clusters() << std::endl;
}

TEST(SpectralNorm, SVDvsDiagonalization) {
  typedef std::complex<double> Scalar;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mat(2, 6);
  mat.fill(0.0);
  mat(0, 0) = std::complex<double>(-1, -3.41908e-05);
  mat(0, 1) = std::complex<double>(5.90659e-05, 6.05984e-06);
  mat(0, 2) = std::complex<double>(0.00992848, -6.10314e-05);
  mat(0, 3) = std::complex<double>(0.00305649, 0.000657586);
  mat(0, 4) = std::complex<double>(0.000676405, -5.48592e-06);
  mat(0, 5) = std::complex<double>(-0.00231812, 2.09944e-05);
  mat(1, 5) = std::complex<double>(-1.00231812, 2.09944e-05);

  //std::cout << spectral_norm_SVD<Scalar>(mat) << std::endl;
  //std::cout << spectral_norm_diag<Scalar>(mat) << std::endl;
  ASSERT_TRUE(std::abs(spectral_norm_SVD<Scalar>(mat) - spectral_norm_diag<Scalar>(mat)) < 1E-8);
}

TEST(FastUpdate, CombSort) {
  const int N = 1000;
  std::vector<double> data(N);

  boost::random::mt19937 gen(100);
  boost::uniform_real<> uni_dist(0, 1);

  for (int i = 0; i < N; ++i) {
    data[i] = uni_dist(gen);
  }
  std::vector<double> data2 = data;
  const int perm_sign = alps::fastupdate::detail::comb_sort(data.begin(), data.end(), std::less<double>());
  for (int i = 0; i < N - 1; ++i) {
    ASSERT_TRUE(data[i] <= data[i + 1]);
  }

  {
    int count = 0;
    while (true) {
      bool exchanged = false;
      for (int i = 0; i < N - 1; ++i) {
        if (data2[i] > data2[i + 1]) {
          std::swap(data2[i], data2[i + 1]);
          exchanged = true;
          ++count;
        }
      }
      if (!exchanged) {
        break;
      }
    }
    int perm_sign2 = count % 2 == 0 ? 1 : -1;
    ASSERT_TRUE(perm_sign == perm_sign2);
  }
}
