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


TEST(ModelLibrary, AutoPartioning) {
  alps::params par;
  const int sites = 3;
  par["model.sites"] = sites;
  par["model.spins"] = 2;
  par["model.n_tau_hyb"] = 1000;
  typedef std::complex<double> SCALAR;
  const SCALAR tval = 0.0;
  const double onsite_U = 2.0;
  const double JH = 0.1;
  par["model.onsite_U"] = onsite_U;
  par["model.beta"] = 1.0;
  //par["CUTOFF_HAM"] = 1.0e-10;
  //par["ROTATE_F"] = "";

  std::vector<boost::tuple<int, int, int, int, SCALAR> > Uval_list;
  for (int isp = 0; isp < 2; ++isp) {
    for (int isp2 = 0; isp2 < 2; ++isp2) {
      for (int alpha = 0; alpha < sites; ++alpha) {
        Uval_list.push_back(get_tuple<SCALAR>(alpha, alpha, alpha, alpha, isp, isp2, onsite_U, sites));
      }
      for (int alpha = 0; alpha < sites; ++alpha) {
        for (int beta = 0; beta < sites; ++beta) {
          if (alpha == beta) continue;
          Uval_list.push_back(get_tuple<SCALAR>(alpha, beta, alpha, beta, isp, isp2, onsite_U - 2 * JH, sites));
          Uval_list.push_back(get_tuple<SCALAR>(alpha, beta, beta, alpha, isp, isp2, JH, sites));
        }
      }
    }
  }

  std::vector<boost::tuple<int, int, SCALAR> > t_list;
  for (int isp = 0; isp < 2; ++isp) {
    for (int alpha = 0; alpha < sites; ++alpha) {
      for (int beta = 0; beta < sites; ++beta) {
        if (alpha == beta) continue;
        t_list.push_back(boost::make_tuple(alpha + isp * sites, beta + isp * sites, -tval));
      }
    }
  }

  ImpurityModelEigenBasis<SCALAR>::define_parameters(par);
  ImpurityModelEigenBasis<SCALAR> model(par, t_list, Uval_list);
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

/*
TEST(Util, IteratorOverTwoSets) {
  boost::random::mt19937 gen(100);
  boost::uniform_real<> uni_dist(0, 1);

  for (int test = 0; test < 10; ++test) {
    int size1, size2;
    if (test % 4 == 0) {
      std::cout << "calling " << std::endl;
      size1 = static_cast<int>(100 * uni_dist(gen));
      size2 = static_cast<int>(100 * uni_dist(gen));
    } else if (test % 4 == 1) {
      size1 = 0;
      size2 = static_cast<int>(100 * uni_dist(gen));
    } else if (test % 4 == 2) {
      size1 = static_cast<int>(100 * uni_dist(gen));
      size2 = 0;
    } else {
      size1 = 0;
      size2 = 0;
    }
    std::set<double> set1, set2;
    for (int i = 0; i < size1; ++i) {
      const double rnd = uni_dist(gen);
      set1.insert(rnd);
    }
    for (int i = 0; i < size2; ++i) {
      const double rnd = uni_dist(gen);
      set2.insert(rnd);
    }
    std::set<double> set12;
    set12.insert(set1.begin(), set1.end());
    set12.insert(set2.begin(), set2.end());

    TwoSetView<std::set<double> > view(set1, set2);

    std::cout << "A" << set1.size() << " " << set2.size() << std::endl;
    TwoSetView<std::set<double> >::const_iterator it_prime = view.begin();
    int i = 0;
    for (std::set<double>::iterator it = set12.begin(); it != set12.end(); ++it, ++it_prime) {
      std::cout << i << std::endl;
      ASSERT_TRUE(*it == *it_prime);
      ++ i;
    }

    ASSERT_TRUE(it_prime == view.end());

    std::set<double>::iterator it = set12.begin();
    //int i = 0;
    for (TwoSetView<std::set<double> >::const_iterator it_prime = view.begin(); it_prime != view.end();
         ++it_prime, ++it) {
      //std::cout << i << std::endl;
      ASSERT_TRUE(*it == *it_prime);
      //++ i;
    }
  }
}
*/

TEST(G2Measurement, ForkTermToHartreeTerm) {
  double Lambda = 1E+2;
  int max_dim_f = 5;
  int max_dim_b = 5;
  const int niw_sum = 10000;
  const int niw = 10;

  FermionicIRBasis basis_f(Lambda, max_dim_f);
  BosonicIRBasis basis_b(Lambda, max_dim_b);

  auto Tnl_f = basis_f.Tnl(niw_sum);
  auto Tnl_b = basis_b.Tnl(niw_sum);
  const int dim_f = basis_f.dim();
  const int dim_b = basis_b.dim();

  auto Tnl_f_pn = to_Tnl_pn(Tnl_f, alps::gf::statistics::FERMIONIC);
  auto Tnl_b_pn = to_Tnl_pn(Tnl_b, alps::gf::statistics::BOSONIC);

  using dcomplex = std::complex<double>;

  //Compute w(l, l^prime, n) by summing one freq., which will suffer from truncation errors
  Eigen::Tensor<dcomplex,3> w_tensor(niw, dim_b, dim_f);
  auto back_to_range = [](int i) {return std::max(std::min(i, 2*niw_sum),0);};
  for (int lp = 0; lp < dim_f; ++lp) {
    for (int l = 0; l < dim_b; ++l) {
      for (int n = 0; n < niw; ++n) {
        const auto min_m = back_to_range(-n);
        const auto max_m = back_to_range(2*niw_sum -n);
        dcomplex tmp = 0.0;
        for (int m = min_m; m < max_m; ++m) {
          tmp += std::conj(Tnl_b_pn(m,l)) * Tnl_f_pn(m+n, lp);
        }
        w_tensor(n,l,lp) = tmp;
      }
    }
  }

  //Alternatively, use the tau formula, which will give more accurate results.
  Eigen::Tensor<dcomplex,3> w_tensor2;
  compute_w_tensor(niw, basis_f, basis_b, w_tensor2);

  //The two results should match approximately.
  for (int lp=0; lp<dim_f; ++lp) {
    for (int l=0; l<dim_b; ++l) {
      for (int n=0; n<niw; ++n) {
        ASSERT_NEAR(abs(w_tensor(n, l, lp)-w_tensor2(n, l, lp)), 0.0,  0.002);
      }
    }
  }
}

TEST(G2Measurement, ForkTermToHartreeTerm2) {
  using dcomplex = std::complex<double>;

  double Lambda = 1E+1;
  int max_dim_f = 5;
  int max_dim_b = 5;
  const int niw = 1000;

  FermionicIRBasis basis_f(Lambda, max_dim_f);
  BosonicIRBasis basis_b(Lambda, max_dim_b);

  auto dim_f = basis_f.dim();
  auto dim_b = basis_b.dim();

  Eigen::Tensor<dcomplex,6> tensor;
  compute_tensor_from_H_to_F_term(niw, basis_f, basis_b, tensor);

  //Must be a real symmetric matrix
  {
    Eigen::TensorMap<Eigen::Tensor<dcomplex,2>> mat_view(tensor.data(), dim_f*dim_f*dim_b, dim_f*dim_f*dim_b);
    for (int j=0; j < dim_f*dim_f*dim_b; ++j) {
      for (int i=0; i < dim_f*dim_f*dim_b; ++i) {
        ASSERT_NEAR(std::abs(mat_view(i,j) - mat_view(j,i)), 0.0, 1e-8);
      }
    }
  }
}


