/**
 * Fast-update formula based on block matrix representation
 * Copyright (C) 2016 by Hiroshi Shinaoka <h.shinaoka@gmail.com>
 *
 */
#pragma once

#include "resizable_matrix.hpp"
#include "./detail/util.hpp"


namespace alps {
  namespace fastupdate {

    /**
     * Compute the determinant ratio with addition rows and cols
     * We implement equations in Appendix B.1.1 of Luitz's thesis.
     * https://opus.bibliothek.uni-wuerzburg.de/files/6408/thesis_luitz.pdf
     *
     * @param B right top block of the new matrix
     * @param C left bottom block of the new matrix
     * @param D right bottom block of the new matrix
     * @param invA inverse of the currrent matrix
     */
    template<typename Scalar, typename Derived>
    Scalar
      compute_det_ratio_up(
      const Eigen::MatrixBase<Derived> &B,
      const Eigen::MatrixBase<Derived> &C,
      const Eigen::MatrixBase<Derived> &D,
      const ResizableMatrix<Scalar> &invA);

    /**
     * Update the inverse matrix by adding rows and cols
     * We implement equations in Appendix B.1.1 of Luitz's thesis.
     * https://opus.bibliothek.uni-wuerzburg.de/files/6408/thesis_luitz.pdf
     *
     * @param B right top block of the new matrix
     * @param C left bottom block of the new matrix
     * @param D right bottom block of the new matrix
     * @param invA inverse of the currrent matrix. invA is resized automatically.
     */
    template<typename Scalar, typename Derived>
    Scalar
      compute_inverse_matrix_up(
      const Eigen::MatrixBase<Derived> &B,
      const Eigen::MatrixBase<Derived> &C,
      const Eigen::MatrixBase<Derived> &D,
      ResizableMatrix<Scalar> &invA);

    /**
     * Compute the determinant ratio for the removal of rows and cols
     * We implement equations in Appendix B.1.1 of Luitz's thesis.
     * https://opus.bibliothek.uni-wuerzburg.de/files/6408/thesis_luitz.pdf
     *
     * For a certain matrix G, its inverse is denoted by G^{-1}
     * Let us consider removing several rows and columns in G.
     * The resultant matrix is G'.
     * As mentioned below, some of rows and columns in G' are exchanged.
     * In this function, we compute |G'|/|G|, which includes the sign change due to the permutations of rows and columns.
     * Note that swapping rows/columns in a matrix corresponds to
     * swapping the corresponding columns/rows in its inverse, respectively.
     * (G: row <=> G^-1: column)
     *
     * @param num_rows_cols_removed number of rows and cols to be removed in G
     * @param rows_removed positions of rows to be removed in G (not G^{-1}). The first num_rows_cols_removed elements are referred.
     * @param cols_removed positions of cols to be removed in G (not G^{-1}). The first num_rows_cols_removed elements are referred.
     * @param invG inverse of the currrent matrix: G^{-1}
     */
    template<class Scalar>
    Scalar
      compute_det_ratio_down(
      const int num_rows_cols_removed,
      const std::vector<int> &rows_removed,
      const std::vector<int> &cols_removed,
      const ResizableMatrix<Scalar> &invG);

    /**
     * Similar function but just try to remove the last operators.
     */
    template<class Scalar>
    Scalar
      compute_det_ratio_down(
      const int num_rows_cols_removed,
      const ResizableMatrix<Scalar> &invG);

    /**
     * Update the inverse matrix for the removal of rows and cols
     * We implement equations in Appendix B.1.1 of Luitz's thesis.
     * https://opus.bibliothek.uni-wuerzburg.de/files/6408/thesis_luitz.pdf
     *
     * The actual procedure is the following.
     * First, we move all rows and cols to be removed to the last (step1).
     * Then, we remove them (step2).
     * On exit, the positions of some remaining rows and cols are exchanged in step1.
     *
     * @param num_rows_cols_removed number of rows and cols to be removed in G
     * @param rows_removed positions of rows to be removed in G (not G^{-1}). The first num_rows_cols_removed elements are referred.
     * @param cols_removed positions of cols to be removed in G (not G^{-1}). The first num_rows_cols_removed elements are referred.
     * @param invG G^{-1}. invG will be resized and updated to G'^{-1}.
     * @param swapped_rows a list of pairs of rows in G swapped in step 1
     * @param swapped_cols a list of pairs of cols in G swapped in step 1
     */
    template<class Scalar>
    void
      compute_inverse_matrix_down(
      const int num_rows_cols_removed,
      const std::vector<int> &rows_removed,
      const std::vector<int> &cols_removed,
      ResizableMatrix<Scalar> &invG
    );

    template<class Scalar>
    void
      compute_inverse_matrix_down(
      const int num_rows_cols_removed,
      ResizableMatrix<Scalar> &invG
    );

    /**
     * Update the inverse matrix for addition and removal of rows and cols
     * We implement Ye-Hua Lie and Lei Wang (2015): Eqs. (17)-(26) before taking the limit of tS->0
     * http://arxiv.org/abs/1510.00715v2
     *
     * The naming rule is like this: tSp = tilde S^prime
     */
    template<typename Scalar, typename M0, typename M1, typename M2>
    class ReplaceHelper {
    public:
      ReplaceHelper() {};
      ReplaceHelper(ResizableMatrix<Scalar>& invG, const M0& R, const M1& Q, const M2& S);
      Scalar compute_det_ratio(ResizableMatrix<Scalar>& invG, const M0& R, const M1& Q, const M2& S);
      void compute_inverse_matrix(ResizableMatrix<Scalar>& invG, const M0& R, const M1& Q, const M2& S);

    private:
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;
      typedef Eigen::Block<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > block_t;

      //ResizableMatrix<Scalar>& invG_;
      //M0& Q_;
      //M1& R_;
      //M2& S_;
      int N_, M_, M_old_;

      eigen_matrix_t Mmat_, inv_tSp_;
    };


    /**
     * Compute deteterminat ratio for the replacement of the last row of the G matrix
     */
    template<typename Scalar, typename Derived>
    Scalar compute_det_ratio_relace_last_row(const ResizableMatrix<Scalar> & invG,
                                   const Eigen::MatrixBase<Derived>& new_row_elements);

    /**
     * Replace the last row of the G matrix
     */
    template<typename Scalar, typename Derived>
    void compute_inverse_matrix_replace_last_row(ResizableMatrix<Scalar> & invG,
                             const Eigen::MatrixBase<Derived>& new_row_elements, Scalar det_rat);

    /**
     * Compute deteterminat ratio for the replacement of the last column of the G matrix
     */
    template<typename Scalar, typename Derived>
    Scalar compute_det_ratio_relace_last_col(const ResizableMatrix<Scalar> & invG,
                                             const Eigen::MatrixBase<Derived>& new_col_elements);

    /**
     * Replace the last col of the G matrix
     */
    template<typename Scalar, typename Derived>
    void compute_inverse_matrix_replace_last_col(ResizableMatrix<Scalar> & invG,
                                                 const Eigen::MatrixBase<Derived>& new_col_elements, Scalar det_rat);
  }
}

#include "./detail/fastupdate_formula.ipp"
