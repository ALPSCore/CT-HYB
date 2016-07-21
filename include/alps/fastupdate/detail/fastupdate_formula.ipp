/**
 * Fast-update formula based on block matrix representation
 * Copyright (C) 2016 by Hiroshi Shinaoka <h.shinaoka@gmail.com>
 *
 */

#include "../fastupdate_formula.hpp"

/**
 * Some utilities
 */
namespace alps {
  namespace fastupdate {

    template<typename Derived>
    inline int num_cols(const Eigen::MatrixBase<Derived> &m) {
      return m.cols();
    }

    template<typename Derived>
    inline int num_rows(const Eigen::MatrixBase<Derived> &m) {
      return m.rows();
    }
  }
}

/**
 * Definition of fast-update formula for adding rows and cols
 */
namespace alps {
  namespace fastupdate {
    template<typename Scalar, typename Derived>
    Scalar
    compute_det_ratio_up(
      const Eigen::MatrixBase<Derived> &B,
      const Eigen::MatrixBase<Derived> &C,
      const Eigen::MatrixBase<Derived> &D,
      const ResizableMatrix <Scalar> &invA) {
      const size_t N = invA.size1();
      const size_t M = D.rows();

      assert(M > 0);

      assert(num_rows(invA) == num_cols(invA));
      assert(num_rows(B) == N && num_cols(B) == M);
      assert(num_rows(C) == M && num_cols(C) == N);
      assert(num_rows(D) == M && num_cols(D) == M);

      if (N == 0) {
        return detail::safe_determinant(D);
      } else {
        //compute H
        return detail::safe_determinant(D - C * invA.block() * B);
      }
    }

    template<typename Scalar, typename Derived>
    Scalar
    compute_inverse_matrix_up(
      const Eigen::MatrixBase<Derived> &B,
      const Eigen::MatrixBase<Derived> &C,
      const Eigen::MatrixBase<Derived> &D,
      ResizableMatrix <Scalar> &invA) {
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;
      typedef Eigen::Block<eigen_matrix_t> block_t;

      const int N = num_rows(invA);
      const int M = num_rows(D);

      assert(M > 0);
      assert(num_rows(invA) == num_cols(invA));
      assert(num_rows(B) == N && num_cols(B) == M);
      assert(num_rows(C) == M && num_cols(C) == N);
      assert(num_rows(D) == M && num_cols(D) == M);

      if (N == 0) {
        invA = detail::safe_inverse(D);
        return detail::safe_determinant(D);
      } else {
        //I don't know how expensive to allocate temporary objects C_invA, H, invA_B, F.
        //We could keep them as static objects or members of a class.

        //compute H
        const eigen_matrix_t C_invA = C * invA.block();
        const eigen_matrix_t H = detail::safe_inverse(D - C_invA * B);

        //compute F
        const eigen_matrix_t invA_B = invA.block() * B;
        const eigen_matrix_t F = -invA_B * H;

        invA.conservative_resize(N + M, N + M);//this keeps the contents in the left corner of invA

        //compute G
        invA.block(N, 0, M, N).noalias() = -H * C_invA;

        //compute E
        invA.block(0, 0, N, N).noalias() -= invA_B * invA.block(N, 0, M, N);

        invA.block(0, N, N, M) = F;
        invA.block(N, N, M, M) = H;

        return 1. / detail::safe_determinant(H);
      }
    }
  }
}

/**
 * Definition of fast-update formula for removing rows and cols
 */
namespace alps {
  namespace fastupdate {
    template<class Scalar>
    Scalar
    compute_det_ratio_down(
      const int num_rows_cols_removed,
      const std::vector<int> &rows_removed,
      const std::vector<int> &cols_removed,
      const ResizableMatrix <Scalar> &invG) {
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

      const int NpM = num_rows(invG);
      const int M = num_rows_cols_removed;
      assert(num_cols(invG) == NpM);
      assert(rows_removed.size() >= M);
      assert(M > 0);

      //Note: if rows_removed==cols_removed, there is no sign change.
      // Thus, we count the difference from this: perm;
      eigen_matrix_t H(M, M);
      unsigned long perm = 0;
      for (int j = 0; j < M; ++j) {
        perm += std::abs(rows_removed[j] - cols_removed[j]);
        for (int i = 0; i < M; ++i) {
          assert(cols_removed[i] < NpM);
          assert(rows_removed[j] < NpM);
          H(i, j) = invG(cols_removed[i], rows_removed[j]);//Note: rows and cols are inverted.
        }
      }

      return perm % 2 == 0 ? detail::safe_determinant(H) : -detail::safe_determinant(H);
    }

    template<class Scalar>
    Scalar
    compute_det_ratio_down(
      const int num_rows_cols_removed,
      const ResizableMatrix <Scalar> &invG) {
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

      const int NpM = num_rows(invG);
      const int M = num_rows_cols_removed;
      const int N = NpM-M;
      assert(num_cols(invG) == NpM);
      assert(M > 0);

      eigen_matrix_t H(M, M);
      for (int j = 0; j < M; ++j) {
        for (int i = 0; i < M; ++i) {
          H(i, j) = invG(N+i, N+j);
        }
      }

      return detail::safe_determinant(H);
    }

    template<class Scalar>
    void
    compute_inverse_matrix_down(
      const int num_rows_cols_removed,
      const std::vector<int> &rows_removed,
      const std::vector<int> &cols_removed,
      ResizableMatrix <Scalar> &invG
    ) {
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

      const int NpM = num_rows(invG);
      const int M = num_rows_cols_removed;
      const int N = NpM - M;
      assert(num_cols(invG) == NpM);
      assert(rows_removed.size() >= M);
      assert(cols_removed.size() >= M);
      assert(M > 0);
      assert(NpM >= M);

      if (NpM < M) {
        throw std::logic_error("N should not be negative!");
      }

      if (M == 0) {
        throw std::logic_error("M should be larger than 0!");
      }

#ifndef NDEBUG
      //make sure the indices are in ascending order.
      for (int idel = 0; idel < M - 1; ++idel) {
        assert(rows_removed[idel] < rows_removed[idel + 1]);
        assert(cols_removed[idel] < cols_removed[idel + 1]);
      }
#endif

      //Step 1: move rows and cols to be removed to the end.
      for (int idel = 0; idel < M; ++idel) {
        //Note: If we swap two rows in G, this corresponds to swapping the corresponding COLUMNS in G^{-1}
        invG.swap_col(rows_removed[M - 1 - idel], NpM - 1 - idel);
        invG.swap_row(cols_removed[M - 1 - idel], NpM - 1 - idel);
      }

      //Step 2: update the inverse matrix and shrink it.
      if (N > 0) {
        //E -= F*H^{-1}*G
        //(N,M)x(M,M)x(M,N)
        invG.block(0, 0, N, N).noalias() -=
          invG.block(0, N, N, M) *
          detail::safe_inverse(invG.block(N, N, M, M)) *
          invG.block(N, 0, M, N);
      }
      invG.conservative_resize(N, N);
    }

    template<class Scalar>
    void
    compute_inverse_matrix_down(
      const int num_rows_cols_removed,
      ResizableMatrix <Scalar> &invG
    ) {
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

      const int NpM = num_rows(invG);
      const int M = num_rows_cols_removed;
      const int N = NpM - M;
      assert(num_cols(invG) == NpM);
      assert(M > 0);
      assert(NpM >= M);

      if (NpM < M) {
        throw std::logic_error("N should not be negative!");
      }

      if (M == 0) {
        throw std::logic_error("M should be larger than 0!");
      }

      //update the inverse matrix and shrink it.
      if (N > 0) {
        //E -= F*H^{-1}*G
        //(N,M)x(M,M)x(M,N)
        invG.block(0, 0, N, N).noalias() -=
          invG.block(0, N, N, M) *
          detail::safe_inverse(invG.block(N, N, M, M)) *
          invG.block(N, 0, M, N);
      }
      invG.conservative_resize(N, N);
    }
  }
}

/**
 * Definition of ReplaceHelper
 */
namespace alps {
  namespace fastupdate {

    template<typename Scalar, typename M0, typename M1, typename M2>
    ReplaceHelper<Scalar,M0,M1,M2>::ReplaceHelper(ResizableMatrix<Scalar>& invG,
                                                  const M0& Q,
                                                  const M1& R,
                                                  const M2& S) :
      //invG_(invG),
      //Q_(Q),
      //R_(R),
      //S_(S),
      N_(num_cols(R)),
      M_(num_rows(R)),
      M_old_(num_cols(invG)-N_)
    {
      assert(num_cols(invG)==num_rows(invG));
      assert(num_rows(R)==M_ && num_cols(R)==N_);
      assert(num_rows(Q)==N_ && num_cols(Q)==M_);
      assert(num_rows(S)==M_ && num_cols(S)==M_);
    }

    template<typename Scalar, typename M0, typename M1, typename M2>
    Scalar ReplaceHelper<Scalar,M0,M1,M2>::compute_det_ratio(ResizableMatrix<Scalar>& invG,
                                                             const M0& Q,
                                                             const M1& R,
                                                             const M2& S) {
      if (N_ == 0) {
        return detail::safe_determinant(S)*invG.safe_determinant();
      }

      block_t tP_view (invG.block(0,  0,  N_,     N_     ));
      block_t tQ_view (invG.block(0,  N_, N_,     M_old_ ));
      block_t tR_view (invG.block(N_, 0,  M_old_, N_     ));
      block_t tS_view (invG.block(N_, N_, M_old_, M_old_ ));

      //FIX ME: use an alternative formula for the case the intermediate state is singular
      if (detail::safe_determinant(tS_view) == 0.0) {
        std::cout << "Warning: intermediate state is singular " << std::endl;
        return 0.0;
      }

      //matrix M
      Mmat_ = tP_view;
      if (M_old_ > 0) {
        Mmat_.noalias() -= tQ_view * detail::safe_inverse(tS_view) * tR_view; //(N, M_old) x (M_old, M_old) x (M_old, N)
      }

      //(tS')^{-1}
      inv_tSp_ = S;
      inv_tSp_.noalias() -= R * (Mmat_ * Q).eval(); //(M,N)x(N,N)x(N,M)

      return detail::safe_determinant(tS_view)*detail::safe_determinant(inv_tSp_);
    }

    template<typename Scalar, typename M0, typename M1, typename M2>
    void ReplaceHelper<Scalar,M0,M1,M2>::compute_inverse_matrix(ResizableMatrix<Scalar>& invG,
                                                                const M0& Q,
                                                                const M1& R,
                                                                const M2& S) {
      if (N_ == 0) {
        invG.destructive_resize(M_, M_);
        if(M_ > 0) {
          invG.block().noalias() = detail::safe_inverse(S);
        }
        return;
      }

      invG.destructive_resize(N_ + M_, N_ + M_);
      block_t tPp_view(invG.block(0,  0,  N_, N_));
      block_t tQp_view(invG.block(0,  N_, N_, M_));
      block_t tRp_view(invG.block(N_, 0,  M_, N_));
      block_t tSp_view(invG.block(N_, N_, M_, M_));

      if (M_ > 0) {
        //tSp
        tSp_view.noalias() = detail::safe_inverse(inv_tSp_);

        //tQp
        //(N,N)x(N,M)x(M,M) = (N,M)
        tQp_view.noalias() = -Mmat_ * (Q * tSp_view).eval();

        //tRp
        //(M,M)x(M,N)x(N,N)
        tRp_view.noalias() = -(tSp_view * R).eval() * Mmat_;
      }

      //tPp
      tPp_view = Mmat_;
      if (M_ > 0) {
        tPp_view -= (Mmat_ * Q).eval() * tRp_view; //(N,N)x(N,M)x(M,N)
      }
    }

    /**
     * Compute deteterminat ratio for the replacement of the last row of the G matrix
     */
    template<typename Scalar, typename Derived>
    Scalar compute_det_ratio_relace_last_row(const ResizableMatrix<Scalar> & invG,
                                             const Eigen::MatrixBase<Derived>& new_row_elements) {
      assert(new_row_elements.rows()==1);
      Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> r(1,1);
      r = new_row_elements*invG.block().col(invG.size2()-1);
      return r(0,0);
    }

    /**
     * Replace the last row of the G matrix
     */
    template<typename Scalar, typename Derived>
    void compute_inverse_matrix_replace_last_row(ResizableMatrix<Scalar> & invG,
                                                 const Eigen::MatrixBase<Derived>& new_row_elements, Scalar det_rat) {
      assert(new_row_elements.rows()==1);

      const int N = invG.size1();
      Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> R(1, N), last_col(N,1);
      last_col = invG.block().col(N-1);

      R(0, N-1) = 0.0;
      R.block(0,0, 1,N-1) = new_row_elements*invG.block(0,0, N,N-1);
      const Scalar inv_det_rat = 1.0/det_rat;
      invG.block().col(N-1) = last_col*inv_det_rat;

      for (int m = 0; m < N-1; m++) {
        for (int n = 0; n < N; n++) {
          invG(n, m) -= last_col(n,0)*R(0,m)*inv_det_rat;
        }
      }
    }

    /**
     * Compute deteterminat ratio for the replacement of the last column of the G matrix
     */
    template<typename Scalar, typename Derived>
    Scalar compute_det_ratio_relace_last_col(const ResizableMatrix<Scalar> & invG,
                                             const Eigen::MatrixBase<Derived>& new_col_elements) {
      assert(new_col_elements.cols()==1);
      Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> r(1,1);
      r = invG.block().row(invG.size2()-1)*new_col_elements;
      return r(0,0);
    }

    /**
     * Replace the last col of the G matrix
     */
    template<typename Scalar, typename Derived>
    void compute_inverse_matrix_replace_last_col(ResizableMatrix<Scalar> & invG,
                                                 const Eigen::MatrixBase<Derived>& new_col_elements, Scalar det_rat) {
      assert(new_col_elements.cols()==1);

      const int N = invG.size1();
      Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> R(N, 1), last_row(1,N);
      last_row = invG.block().row(N-1);

      R(N-1, 0) = 0.0;
      R.block(0,0, N-1,1) = invG.block(0,0, N-1,N)*new_col_elements;

      const Scalar inv_det_rat = 1.0/det_rat;
      invG.block().row(N-1) = last_row*inv_det_rat;

      for (int m = 0; m < N-1; m++) {
        for (int n = 0; n < N; n++) {
          invG(m, n) -= last_row(0,n)*R(m,0)*inv_det_rat;
        }
      }

    }
  }
}
