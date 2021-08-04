#include "unittest_fu.hpp"


TEST(FastUpdate, BlockMatrixAdd)
{
  using namespace alps::fastupdate;

  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

  std::vector<size_t> N_list, M_list;
  N_list.push_back(0);
  N_list.push_back(10);
  N_list.push_back(2);

  M_list.push_back(10);
  M_list.push_back(20);
  M_list.push_back(4);

  for (size_t n=0; n<N_list.size(); ++n) {
    for (size_t m=0; m<M_list.size(); ++m) {
      const size_t N = N_list[n];
      const size_t M = M_list[m];

      eigen_matrix_t A(N,N), B(N,M), C(M,N), D(M,M);
      eigen_matrix_t E(N,N), F(N,M), G(M,N), H(M,M);
      ResizableMatrix<Scalar> invA(N,N), BigMatrix(N+M, N+M, 0);//, invBigMatrix2(N+M, N+M, 0);

      randomize_matrix(A, 100);//100 is a seed
      randomize_matrix(B, 200);
      randomize_matrix(C, 300);
      randomize_matrix(D, 400);
      if (N>0) {
        invA = A.inverse();
      } else {
        invA.destructive_resize(0,0);
      }

      copy_block(A,0,0,BigMatrix,0,0,N,N);
      copy_block(B,0,0,BigMatrix,0,N,N,M);
      copy_block(C,0,0,BigMatrix,N,0,M,N);
      copy_block(D,0,0,BigMatrix,N,N,M,M);

      const Scalar det_rat = compute_det_ratio_up<Scalar>(B, C, D, invA);
      ASSERT_TRUE(std::abs(det_rat-determinant(BigMatrix)/A.determinant())<1E-8)
                << "N=" << N << " M=" << M << " " << std::abs(det_rat-determinant(BigMatrix)) << "/" << std::abs(det_rat)<<"="
                << std::abs(det_rat-determinant(BigMatrix)/A.determinant());

      const Scalar det_rat2 = alps::fastupdate::compute_inverse_matrix_up(B, C, D, invA);
      ASSERT_TRUE(std::abs(det_rat-det_rat2)<1E-8) << "N=" << N << " M=" << M;
      ASSERT_TRUE(norm_square(inverse(BigMatrix)-invA)<1E-8) << "N=" << N << " M=" << M;
    }
  }
}


void select_rows_removed(unsigned int seed, int N, int M, std::vector<int>& rows_removed, std::vector<int>& rows_remain) {
  boost::mt19937 gen(seed);
  rows_removed.resize(N+M);
  rows_remain.resize(N);
  for (int i=0; i<N+M; ++i) {
    rows_removed[i] = i;
  }
  rs_shuffle rs(gen);
  std::random_shuffle(rows_removed.begin(), rows_removed.end(), rs);
  for (int i=0; i<N; ++i) {
    rows_remain[i] = rows_removed[i+M];
  }
  rows_removed.resize(M);
  std::sort(rows_removed.begin(), rows_removed.end());
  std::sort(rows_remain.begin(), rows_remain.end());
}

TEST(FastUpdate, BlockMatrixDown)
{
  using namespace alps::fastupdate;
  typedef double Scalar;

  std::vector<int> N_list, M_list;
  N_list.push_back(10);
  M_list.push_back(10);
  M_list.push_back(20);
  M_list.push_back(30);

  for (int n=0; n<N_list.size(); ++n) {
    for (int m=0; m<M_list.size(); ++m) {
      const int N = N_list[n];
      const int M = M_list[m];

      ResizableMatrix<Scalar> G(N+M, N+M, 0.0), invG(N+M, N+M, 0.0);//G, G^{-1}
      ResizableMatrix<Scalar> Gprime(N, N, 0.0);//G'

      randomize_matrix(G, 100);//100 is a seed
      invG = inverse(G);

      //which rows and cols are to be removed
      std::vector<int> rows_removed(N+M), rows_remain(N), cols_removed(N+M), cols_remain(N);
      select_rows_removed(919, N, M, rows_removed, rows_remain);
      select_rows_removed(119, N, M, cols_removed, cols_remain);

      for (int j=0; j<N; ++j) {
        for (int i=0; i<N; ++i) {
          Gprime(i,j) = G(rows_remain[i], cols_remain[j]);
        }
      }

      //testing compute_det_ratio_down
      Scalar det_rat = compute_det_ratio_down(M, rows_removed, cols_removed, invG);
      ASSERT_TRUE(std::abs(det_rat-Gprime.determinant()/G.determinant())<1E-8) << "N=" << N << " M=" << M;

      //update G^{-1} to G'^{-1}
      ResizableMatrix<Scalar> invGprime_fastupdate(invG);
      compute_inverse_matrix_down(M, rows_removed, cols_removed, invGprime_fastupdate);

      //Note that remaining rows and cols may be swapped in the fastupdate
      ResizableMatrix<Scalar> invGprime = G;
      for (int s=0; s<M; ++s) {
        invGprime.swap_row(rows_removed[M-1-s], N+M-1-s);
        invGprime.swap_col(cols_removed[M-1-s], N+M-1-s);
      }
      invGprime.conservative_resize(N,N);
      invGprime.invert();
      ASSERT_TRUE(norm_square(invGprime-invGprime_fastupdate)<1E-8) << "N=" << N << " M=" << M;
    }
  }
}

TEST(FastUpdate, ReplaceLastRow)
{
  using namespace alps::fastupdate;
  typedef std::complex<double> Scalar;

  std::vector<int> N_list;
  N_list.push_back(10);
  N_list.push_back(21);

  for (int n=0; n<N_list.size(); ++n) {
    const int N = N_list[n];

    ResizableMatrix<Scalar> G(N, N, 0.0), invG(N, N, 0.0);//G, G^{-1}
    ResizableMatrix<Scalar> Gprime(N, N, 0.0);//G'
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> new_row(1, N);

    randomize_matrix(G, 100);//100 is a seed
    invG = inverse(G);

    randomize_matrix(new_row, 101);//100 is a seed
    Gprime = G;
    for (int j=0; j<N; ++j) {
      Gprime(N-1, j) = new_row(0, j);
    }

    Scalar det_rat = compute_det_ratio_relace_last_row(invG, new_row);
    ASSERT_TRUE(std::abs(det_rat-Gprime.determinant()/G.determinant())<1E-8) << "N=" << N;

    //update G^{-1} to G'^{-1}
    ResizableMatrix<Scalar> invGprime_fastupdate(invG);
    compute_inverse_matrix_replace_last_row(invG, new_row, det_rat);

    ResizableMatrix<Scalar> invGprime = G;
    invGprime.invert();
    ASSERT_TRUE(norm_square(invGprime-invGprime_fastupdate)<1E-8) << "N=" << N;
  }
}

TEST(FastUpdate, ReplaceLastCol)
{
  using namespace alps::fastupdate;
  typedef std::complex<double> Scalar;

  std::vector<int> N_list;
  N_list.push_back(10);
  N_list.push_back(21);

  for (int n=0; n<N_list.size(); ++n) {
    const int N = N_list[n];

    ResizableMatrix<Scalar> G(N, N, 0.0), invG(N, N, 0.0);//G, G^{-1}
    ResizableMatrix<Scalar> Gprime(N, N, 0.0);//G'
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> new_col(N, 1);

    randomize_matrix(G, 100);//100 is a seed
    invG = inverse(G);

    randomize_matrix(new_col, 101);//100 is a seed
    Gprime = G;
    for (int j=0; j<N; ++j) {
      Gprime(j, N-1) = new_col(j, 0);
    }

    Scalar det_rat = compute_det_ratio_relace_last_col(invG, new_col);
    ASSERT_TRUE(std::abs(det_rat-Gprime.determinant()/G.determinant())<1E-8) << "N=" << N;

    //update G^{-1} to G'^{-1}
    ResizableMatrix<Scalar> invGprime_fastupdate(invG);
    compute_inverse_matrix_replace_last_col(invG, new_col, det_rat);

    ResizableMatrix<Scalar> invGprime = G;
    invGprime.invert();
    ASSERT_TRUE(norm_square(invGprime-invGprime_fastupdate)<1E-8) << "N=" << N;
  }
}

TEST(FastUpdate, BlockMatrixReplaceLastRowsColsWithDifferentSizes) {
  using namespace alps::fastupdate;
  typedef std::complex<double> Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;
  typedef ReplaceHelper<Scalar, eigen_matrix_t, eigen_matrix_t, eigen_matrix_t> helper_t;

  std::vector<int> N_list, M_list, Mold_list;
  N_list.push_back(0);
  M_list.push_back(4);
  Mold_list.push_back(5);

  N_list.push_back(2);
  M_list.push_back(0);
  Mold_list.push_back(5);

  N_list.push_back(2);
  M_list.push_back(5);
  Mold_list.push_back(0);

  N_list.push_back(100);
  M_list.push_back(40);
  Mold_list.push_back(50);

  N_list.push_back(100);
  M_list.push_back(49);
  Mold_list.push_back(20);

  //N_list.push_back(100);
  //M_list.push_back(100);
  //Mold_list.push_back(20);

  for (int n = 0; n < N_list.size(); ++n) {
    for (int m = 0; m < M_list.size(); ++m) {
      const int N = N_list[n];
      const int M = M_list[m];
      const int Mold = Mold_list[m];

      ResizableMatrix<Scalar> G(N + Mold, N + Mold, 0), invG(N + Mold, N + Mold, 0);

      randomize_matrix(G, 100);//100 is a seed
      invG = inverse(G);

      //New entries
      eigen_matrix_t R(M, N), S(M, M), Q(N, M);
      randomize_matrix(R, 110);//100 is a seed
      randomize_matrix(S, 210);//100 is a seed
      randomize_matrix(Q, 310);//100 is a seed

      ResizableMatrix<Scalar> Gprime(N+M, N+M);
      copy_block(G, 0, 0, Gprime, 0, 0, N, N);
      copy_block(R, 0, 0, Gprime, N, 0, M, N);
      copy_block(Q, 0, 0, Gprime, 0, N, N, M);
      copy_block(S, 0, 0, Gprime, N, N, M, M);

      //testing compute_det_ratio_down
      const Scalar det_rat = determinant(Gprime)/determinant(G);

      // construct a helper
      ResizableMatrix<Scalar> invGprime_fast(invG);//, Mmat, inv_tSp, tPp, tQp, tRp, tSp;
      helper_t helper(invGprime_fast, Q, R, S);

      // compute det ratio
      const Scalar det_rat_fast = helper.compute_det_ratio(invGprime_fast, Q, R, S);
      ASSERT_TRUE(std::abs(det_rat-det_rat_fast)/std::abs(det_rat)<1E-8);

      // update the inverse matrix
      helper.compute_inverse_matrix(invGprime_fast, Q, R, S);

      ASSERT_TRUE(norm_square(inverse(Gprime)-invGprime_fast)<1E-8);
    }
  }
}

template<class T>
class DeterminantMatrixTypedTest : public testing::Test {
};
TYPED_TEST_CASE(DeterminantMatrixTypedTest , TestTypes);

TYPED_TEST(DeterminantMatrixTypedTest, CombinedUpdateRemoveRowsCols) {
  using namespace alps::fastupdate;
  typedef std::complex<double> Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

  const int n_flavors = 4;
  const double beta = 1.0;
  typedef TypeParam determinant_matrix_t;
  const int seed = 122;
  boost::mt19937 gen(seed);
  boost::uniform_01<> unidist;
  rs_shuffle rs(gen);

  std::vector<double> E(n_flavors);
  boost::multi_array<Scalar,2> phase(boost::extents[n_flavors][n_flavors]);

  for (int i=0; i<n_flavors; ++i) {
    E[i] = 0.001;
  }
  for (int i=0; i<n_flavors; ++i) {
    for (int j=i; j<n_flavors; ++j) {
      phase[i][j] = std::exp(std::complex<double>(0.0, 1.*i*(2*j+1.0)));
      phase[j][i] = std::conj(phase[i][j]);
    }
  }


  /*
  std::vector<std::pair<creator,annihilator> > init_ops;
  for (int i=0; i<4; ++i) {
    const int f1 = n_flavors*unidist(gen);
    const int f2 = n_flavors*unidist(gen);
    const double t1 = unidist(gen)*beta;
    const double t2 = unidist(gen)*beta;
    init_ops.push_back(
      std::make_pair(
        creator(f1, t1),
        annihilator(f2, t2)
      )
    );
  }
   */

  //OffDiagonalG0<Scalar> gf(beta, n_flavors, E, phase);
  //determinant_matrix_t det_mat(gf, init_ops.begin(), init_ops.end());
  determinant_matrix_t det_mat(
    std::shared_ptr<OffDiagonalG0<Scalar> >(
     new OffDiagonalG0<Scalar>(beta, n_flavors, E, phase)
    )
  );

  const Scalar det_init = det_mat.compute_determinant();

  /*
   * Now we remove some operators
   */
  for (int itest=0; itest<50; ++itest) {
    const Scalar det_old = det_mat.compute_determinant();

    const int num_rem = static_cast<int>(unidist(gen) * det_mat.size());

    std::vector<creator> cdagg_ops = det_mat.get_cdagg_ops();
    std::random_shuffle(cdagg_ops.begin(), cdagg_ops.end(), rs);
    cdagg_ops.resize(num_rem);

    std::vector<annihilator> c_ops = det_mat.get_c_ops();
    std::random_shuffle(c_ops.begin(), c_ops.end(), rs);
    c_ops.resize(num_rem);

    std::vector<creator> cdagg_rem;
    std::vector<annihilator> c_rem;
    for (int iop = 0; iop < num_rem; ++iop) {
      cdagg_rem.push_back(cdagg_ops[iop]);
      c_rem.push_back(c_ops[iop]);
    }

    const int num_add = static_cast<int>(unidist(gen) * 10);
    std::vector<creator> cdagg_add;
    std::vector<annihilator> c_add;
    for (int i = 0; i < num_add; ++i) {
      const int f1 = n_flavors * unidist(gen);
      const int f2 = n_flavors * unidist(gen);
      const double t1 = unidist(gen) * beta;
      const double t2 = unidist(gen) * beta;
      cdagg_add.push_back(creator(f1, t1));
      c_add.push_back(annihilator(f2, t2));
    }

    if (num_rem == 0 || num_add == 0) continue;

    const Scalar det_rat_fast_update = det_mat.try_update(
      cdagg_rem.begin(), cdagg_rem.end(),
      c_rem.begin(),     c_rem.end(),
      cdagg_add.begin(), cdagg_add.end(),
      c_add.begin(),     c_add.end()
    );

    const bool singular =  (std::abs(det_rat_fast_update) < 1E-5 || std::abs(det_rat_fast_update) > 1E+5);

    if (std::abs(det_rat_fast_update) > unidist(gen) && !singular) {
      det_mat.perform_update();
      const Scalar det_new = det_mat.compute_determinant();

      ASSERT_TRUE(std::abs(det_new/det_old-det_rat_fast_update) / std::abs(det_rat_fast_update) < 1E-8);

    } else {
      det_mat.reject_update();
    }

    //check inverse matrix
    eigen_matrix_t inv_mat = det_mat.compute_inverse_matrix();
    det_mat.rebuild_inverse_matrix();
    eigen_matrix_t inv_mat_rebuilt = det_mat.compute_inverse_matrix();
    if (det_mat.size() > 0) {
      ASSERT_TRUE((inv_mat-inv_mat_rebuilt).squaredNorm()/inv_mat_rebuilt.squaredNorm() < 1E-8);
    }
  }
}

TYPED_TEST(DeterminantMatrixTypedTest, SeparatedUpdateRemoveRowsCols) {
  using namespace alps::fastupdate;
  typedef std::complex<double> Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

  const int n_flavors = 4;
  const double beta = 1.0;
  typedef TypeParam determinant_matrix_t;
  const int seed = 122;
  boost::mt19937 gen(seed);
  boost::uniform_01<> unidist;
  rs_shuffle rs(gen);

  std::vector<double> E(n_flavors);
  boost::multi_array<Scalar,2> phase(boost::extents[n_flavors][n_flavors]);

  for (int i=0; i<n_flavors; ++i) {
    E[i] = 0.001;
  }
  for (int i=0; i<n_flavors; ++i) {
    for (int j=i; j<n_flavors; ++j) {
      phase[i][j] = std::exp(std::complex<double>(0.0, 1.*i*(2*j+1.0)));
      phase[j][i] = std::conj(phase[i][j]);
    }
  }

  /*
  std::vector<std::pair<creator,annihilator> > init_ops;
  for (int i=0; i<4; ++i) {
    const int f1 = n_flavors*unidist(gen);
    const int f2 = n_flavors*unidist(gen);
    const double t1 = unidist(gen)*beta;
    const double t2 = unidist(gen)*beta;
    init_ops.push_back(
      std::make_pair(
        creator(f1, t1),
        annihilator(f2, t2)
      )
    );
  }
   */

  //OffDiagonalG0<Scalar> gf(beta, n_flavors, E, phase);
  //determinant_matrix_t det_mat(gf, init_ops.begin(), init_ops.end());
  determinant_matrix_t det_mat(
      std::shared_ptr<OffDiagonalG0<Scalar> >(
          new OffDiagonalG0<Scalar>(beta, n_flavors, E, phase)
      )
  );

  //std::cout << "initial pert " << det_mat.size() << std::endl;

  const Scalar det_init = det_mat.compute_determinant();

  /*
   * Now we remove or add some operators
   */
  for (int itest=0; itest<2000; ++itest) {
    const Scalar det_old = det_mat.compute_determinant();

    int num_rem, num_add;
    const int pert_order = det_mat.size();

    if (unidist(gen) < 0.5 || pert_order==0) {
      num_rem = 0;
      num_add = static_cast<int>(unidist(gen) * 10);
    } else {
      num_rem = static_cast<int>(unidist(gen) * std::min(det_mat.size(),10));
      num_add = 0;
    }

    std::vector<creator> cdagg_ops = det_mat.get_cdagg_ops();
    std::random_shuffle(cdagg_ops.begin(), cdagg_ops.end(), rs);
    cdagg_ops.resize(num_rem);

    std::vector<annihilator> c_ops = det_mat.get_c_ops();
    std::random_shuffle(c_ops.begin(), c_ops.end(), rs);
    c_ops.resize(num_rem);

    std::vector<creator> cdagg_add;
    std::vector<annihilator> c_add;
    for (int i = 0; i < num_add; ++i) {
      const int f1 = n_flavors * unidist(gen);
      const int f2 = n_flavors * unidist(gen);
      const double t1 = unidist(gen) * beta;
      const double t2 = unidist(gen) * beta;
      cdagg_add.push_back(creator(f1, t1));
      c_add.push_back(annihilator(f2, t2));
    }

    if (num_rem == 0 && num_add == 0) continue;

    //std::cout << "pert " << det_mat.size() << " " << num_rem << " " << num_add << std::endl;

    Scalar det_rat_fast_update;
    det_rat_fast_update = det_mat.try_update(
      cdagg_ops.begin(), cdagg_ops.end(),
      c_ops.begin(),     c_ops.end(),
      cdagg_add.begin(), cdagg_add.end(),
      c_add.begin(),     c_add.end()
    );

    const bool singular =  std::abs(det_rat_fast_update) < 1E-5;

    const int pert_order0 = 20;
    const int new_pert_order = pert_order + num_add - num_rem;
    const double p = std::exp(
      -(new_pert_order+pert_order-2*pert_order0)*(new_pert_order-pert_order)/5.0
    );
    if (std::abs(det_rat_fast_update)*p > unidist(gen) && !singular) {
      det_mat.perform_update();
      const Scalar det_new = det_mat.compute_determinant();
      ASSERT_TRUE(std::abs(det_new/det_old-det_rat_fast_update) / std::abs(det_rat_fast_update) < 1E-8);
    } else {
      det_mat.reject_update();
    }

    //check inverse matrix
    eigen_matrix_t inv_mat = det_mat.compute_inverse_matrix();
    det_mat.rebuild_inverse_matrix();
    eigen_matrix_t inv_mat_rebuilt = det_mat.compute_inverse_matrix();
    if (det_mat.size()>0) {
      ASSERT_TRUE((inv_mat-inv_mat_rebuilt).squaredNorm()/inv_mat_rebuilt.squaredNorm() < 1E-8);
    }
  }
}

TYPED_TEST(DeterminantMatrixTypedTest, ReplaceRowCol) {
  using namespace alps::fastupdate;
  typedef std::complex<double> Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

  const int n_flavors = 2;
  const double beta = 1.0;
  const int pert_order = 8;

  typedef TypeParam determinant_matrix_t;
  const int seed = 123;
  boost::mt19937 gen(seed);
  boost::uniform_01<> unidist;
  rs_shuffle rs(gen);

  std::vector<double> E(n_flavors);
  boost::multi_array<Scalar,2> phase(boost::extents[n_flavors][n_flavors]);

  for (int i=0; i<n_flavors; ++i) {
    E[i] = 0.001;
  }
  for (int i=0; i<n_flavors; ++i) {
    for (int j=i; j<n_flavors; ++j) {
      phase[i][j] = std::exp(std::complex<double>(0.0, 1.*i*(2*j+1.0)));
      phase[j][i] = std::conj(phase[i][j]);
    }
  }

  std::vector<std::pair<creator,annihilator> > init_ops;
  for (int i=0; i<pert_order; ++i) {
    const int f1 = n_flavors*unidist(gen);
    const double t1 = unidist(gen)*beta;
    const double t2 = unidist(gen)*beta;
    init_ops.push_back(
      std::make_pair(
        creator(f1, t1),
        annihilator(f1, t2)
      )
    );
  }

  //OffDiagonalG0<Scalar> gf(beta, n_flavors, E, phase);
  determinant_matrix_t det_mat(
      std::shared_ptr<OffDiagonalG0<Scalar> >(
          new OffDiagonalG0<Scalar>(beta, n_flavors, E, phase)
      ),
      init_ops.begin(),
      init_ops.end()
  );

  const Scalar det_init = det_mat.compute_determinant();

  /*
   * Now we place cdagger
   */
  for (int itest=0; itest<100; ++itest) {
    const Scalar det_old = det_mat.compute_determinant();

    std::vector<creator> cdagg_ops = det_mat.get_cdagg_ops();
    std::random_shuffle(cdagg_ops.begin(), cdagg_ops.end(), rs);
    cdagg_ops.resize(1);
    creator new_cdagg(n_flavors*unidist(gen), unidist(gen)*beta);
    const Scalar det_rat_fast_update =
      det_mat.try_update(
        &cdagg_ops[0],       &cdagg_ops[0]+1,
        (annihilator*)NULL,  (annihilator*)NULL,
        &new_cdagg,          &new_cdagg+1,
        (annihilator*)NULL,  (annihilator*)NULL
      );

    const bool singular =  std::abs(det_rat_fast_update) < 1E-5 || std::abs(det_rat_fast_update) > 1E+5;

    if (std::abs(det_rat_fast_update) > unidist(gen) && !singular) {
      det_mat.perform_update();
      const Scalar det_new = det_mat.compute_determinant();
      //std::cout << det_new/det_old << " " << det_rat_fast_update << std::endl;
      ASSERT_TRUE(std::abs(det_new/det_old-det_rat_fast_update) / std::abs(det_rat_fast_update) < 1E-8);
    } else {
      det_mat.reject_update();
    }

    //check inverse matrix
    eigen_matrix_t inv_mat = det_mat.compute_inverse_matrix();
    det_mat.rebuild_inverse_matrix();
    eigen_matrix_t inv_mat_rebuilt = det_mat.compute_inverse_matrix();
    //std::cout << "inv_mat " << std::endl << inv_mat << std::endl;
    //std::cout << "inv_mat_rebuilt " << std::endl << inv_mat_rebuilt << std::endl;
    ASSERT_TRUE((inv_mat-inv_mat_rebuilt).squaredNorm()/inv_mat_rebuilt.squaredNorm() < 1E-8);
  }

  /*
   * Now we place c
   */
  for (int itest=0; itest<100; ++itest) {
    const Scalar det_old = det_mat.compute_determinant();

    std::vector<annihilator> c_ops = det_mat.get_c_ops();
    std::random_shuffle(c_ops.begin(), c_ops.end(), rs);
    c_ops.resize(1);
    annihilator new_c(n_flavors*unidist(gen), unidist(gen)*beta);
    const Scalar det_rat_fast_update =
      det_mat.try_update(
        (creator*)NULL, (creator*)NULL,
        &c_ops[0], &c_ops[0]+1,
        (creator*)NULL, (creator*)NULL,
        &new_c, &new_c+1
      );

    const bool singular =  std::abs(det_rat_fast_update) < 1E-5 || std::abs(det_rat_fast_update) > 1E+5;

    if (std::abs(det_rat_fast_update) > unidist(gen) && !singular) {
      det_mat.perform_update();
      const Scalar det_new = det_mat.compute_determinant();
      const std::vector<Scalar>& det_vec = det_mat.compute_determinant_as_product();
      const Scalar det_new_prod = std::accumulate(det_vec.begin(), det_vec.end(), (Scalar)1.0, std::multiplies<Scalar>());
      ASSERT_TRUE(std::abs(det_new/det_old-det_rat_fast_update) / std::abs(det_rat_fast_update) < 1E-8);
      ASSERT_TRUE(std::abs(det_new_prod/det_old-det_rat_fast_update) / std::abs(det_rat_fast_update) < 1E-8);
    } else {
      det_mat.reject_update();
    }

    //check inverse matrix
    eigen_matrix_t inv_mat = det_mat.compute_inverse_matrix();
    det_mat.rebuild_inverse_matrix();
    eigen_matrix_t inv_mat_rebuilt = det_mat.compute_inverse_matrix();
    ASSERT_TRUE((inv_mat-inv_mat_rebuilt).squaredNorm()/inv_mat_rebuilt.squaredNorm() < 1E-8);
  }

}
