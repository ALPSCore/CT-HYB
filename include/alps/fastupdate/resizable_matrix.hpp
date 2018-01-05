/*
 * Resizable matrix class based on Eigen3
 *  Copyright (C) 2016 by Hiroshi Shinaoka <h.shinaoka@gmail.com>,
 *
 *  based on an earlier version by Emanuel Gull.
 *  This matrix can have a memory size different from the current size to
 *  avoid reallocating memory frequently.
 */
#pragma once

#include<Eigen/Dense>
#include<Eigen/LU>
#include "./detail/util.hpp"

#define ALPS_STRONG_INLINE inline

namespace alps {
  namespace fastupdate {

    template<typename Scalar>
    class ResizableMatrix {
    public:
      typedef Scalar type;
      typedef Eigen::Block<Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > block_type;

      ResizableMatrix(int size1, int size2) :
        size1_(size1),
        size2_(size2),
        values_(size1, size2) {
        assert(size1 >= 0 && size2 >= 0);
      }

      ResizableMatrix(int size1, int size2, Scalar initial_value) :
        size1_(size1),
        size2_(size2),
        values_(size1, size2) {
        assert(size1 >= 0 && size2 >= 0);
        values_.fill(initial_value);
      }

      ResizableMatrix() : size1_(-1), size2_(-1), values_(0, 0) { }

      ResizableMatrix(const ResizableMatrix<Scalar> &M) {
        if (M.is_allocated()) {
          size1_ = M.size1_;
          size2_ = M.size2_;
          values_ = M.values_;
        } else {
          size1_ = -1;
          size2_ = -1;
        }
      }

      inline bool is_allocated() const {
        return size1_ >= 0 && size2_ >= 0;
      }

      inline Scalar &operator()(const int i, const int j) {
        assert(is_allocated());
        assert(i <= size1_);
        assert(j <= size2_);
        return values_(i, j);
      }

      inline const Scalar &operator()(const int i, const int j) const {
        assert(is_allocated());
        assert(i <= size1_);
        assert(j <= size2_);
        return values_(i, j);
      }

      //ResizableMatrix size
      inline int size1() const {
        assert(is_allocated());
        return size1_;
      }

      inline int size2() const {
        assert(is_allocated());
        return size2_;
      }

      inline int memory_size1() const {
        assert(is_allocated());
        return values_.rows();
      }

      inline int memory_size2() const {
        assert(is_allocated());
        return values_.cols();
      }

      /*
      inline void getrow(int k, double *row) const{
        int one=1;
        blas::dcopy_(&size_, &(values_[k*memory_size_]), &one, row, &one);
      }
      inline void getcol(int k, double *col) const{
        int one=1;
        blas::dcopy_(&size_, &(values_[k]), &memory_size_, col, &one);
      }
      inline void setrow(int k, const double *row){
        int one=1;
        blas::dcopy_(&size_, row, &one, &(values_[k*memory_size_]), &one);
      }
      inline void setcol(int k, const double *col){
        int one=1;
        blas::dcopy_(&size_, col, &one, &(values_[k]), &memory_size_);
      }
      */

      //resize while leaving the old values untouched
      //The new elements are not initialized.
      //If the new size is larger than the memory size, memory is reallocated.
      //In this case, we allocate bit larger memory for avoid further memory allocation.
      inline void conservative_resize(int size1, int size2) {
        if (!is_allocated()) {
          values_.resize(size1, size2);
        } else {
          //Should we consider cache line length?
          if (size1 > memory_size1() || size2 > memory_size2()) {
            values_.conservativeResize(1.2 * size1 + 1, 1.2 * size2 + 1);
          }
        }
        size1_ = size1;
        size2_ = size2;
      }

      //Destructive version of resize()
      inline void destructive_resize(int size1, int size2) {
        if (!is_allocated()) {
          values_.resize(size1, size2);
        } else {
          if (size1 > memory_size1() || size2 > memory_size2()) {
            values_.resize(1.2 * size1 + 1, 1.2 * size2 + 1);
          }
        }
        size1_ = size1;
        size2_ = size2;
      }

      //delete last row and column
      inline void remove_row_col_last() {
        assert(is_allocated());
        assert(size1_ > 0);
        assert(size2_ > 0);
        --size1_;
        --size2_;
      }

      //delete last row
      inline void remove_row_last() {
        assert(is_allocated());
        assert(size1_ > 0);
        --size1_;
      }

      //delete last column
      inline void remove_col_last() {
        assert(is_allocated());
        assert(size2_ > 0);
        --size2_;
      }

      //swap two rows
      inline void swap_row(int r1, int r2) {
        assert(is_allocated());
        assert(r1 < size1_);
        assert(r2 < size1_);
        values_.row(r1).swap(values_.row(r2));
      }

      //swap two columns
      inline void swap_col(int c1, int c2) {
        assert(is_allocated());
        assert(c1 < size2_);
        assert(c2 < size2_);
        values_.col(c1).swap(values_.col(c2));
      }

      //swap two rows and columns
      inline void swap_row_col(int c1, int c2) {
        swap_col(c1, c2);
        swap_row(c1, c2);
      }

      /*
      inline void right_multiply(const std::vector<double> &v1, std::vector<double> &v2) const{ //perform v2[i]=M[ij]v1[j]
        char trans='T';
        double alpha=1., beta=0.;    //no need to multiply a constant or add a vector
        int inc=1;
        blas::dgemv_(&trans, &size_, &size_, &alpha, values_, &memory_size_, &(v1[0]), &inc, &beta, &(v2[0]), &inc);
      }
      */

      void set_to_identity() {
        assert(is_allocated());
        assert(size1_ == size2_);
        if (size1_ > 0) {
          block().setIdentity();
        }
      }

      //This is well-defined for a square matrix
      inline Scalar determinant() const {
        assert(is_allocated());
        assert(size1_ == size2_);

        const int size = size1_;

        //the simple ones...
        if (size == 0) return 1;
        if (size == 1) return operator()(0, 0);
        if (size == 2) return operator()(0, 0) * operator()(1, 1) - operator()(0, 1) * operator()(1, 0);

        return block().determinant();
      }

      //This is well-defined for a square matrix
      inline Scalar safe_determinant() const {
        assert(is_allocated());
        assert(size1_ == size2_);

        const int size = size1_;

        //the simple ones...
        if (size == 0) return 1;
        if (size == 1) return operator()(0, 0);
        if (size == 2) return operator()(0, 0) * operator()(1, 1) - operator()(0, 1) * operator()(1, 0);

        return detail::safe_determinant_eigen_block(block());
      }


      inline void clear() {
        if (is_allocated()) {
          block().setZero();
        }
      }

      inline ResizableMatrix operator-(const ResizableMatrix<Scalar> &M2) const {
        assert(is_allocated());
        assert(size1_ == M2.size1_);
        assert(size2_ == M2.size2_);

        ResizableMatrix Msum(*this);
        Msum.values_.block(0, 0, size1_, size2_) -= M2.values_.block(0, 0, size1_, size2_);
        return Msum;
      }

      template<typename Derived>
      inline const ResizableMatrix &operator=(const Eigen::MatrixBase<Derived> &M2) {
        size1_ = M2.rows();
        size2_ = M2.cols();
        values_ = M2;
        return *this;
      }

      /*
      template<typename Derived>
      inline void operator=(const Eigen::MatrixBase<Derived>& mat) {
        destructive_resize(mat.rows(),mat.cols());
        block() = mat;
      }
      */

      inline Scalar max() const {
        assert(is_allocated());
        return block().maxCoeff();
      }

      inline void swap(ResizableMatrix<Scalar> &M2) throw() {
        std::swap(size1_, M2.size1_);
        std::swap(size2_, M2.size2_);
        values_.swap(M2.values_); //I am not sure if it's inexpensive
      }

      inline void invert() {
        assert(is_allocated());
        assert(size1_ == size2_);
        if (is_allocated() && size1_*size2_ > 0) {
          eigen_matrix_t inv = detail::safe_inverse(block());
          values_ = inv;
        }
      }

      inline Scalar trace() {
        assert(is_allocated());
        assert(size1_ == size2_);
        return block().trace();
      }

      ALPS_STRONG_INLINE
      Eigen::Block<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >
      block(int start_row, int start_col, int rows, int cols) {
        assert(is_allocated());
        assert(start_row + rows <= size1());
        assert(start_col + cols <= size2());
        return values_.block(start_row, start_col, rows, cols);
      }

      inline
      Eigen::Block<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >
      block(int start_row, int start_col, int rows, int cols) const {
        assert(is_allocated());
        assert(start_row + rows <= size1());
        assert(start_col + cols <= size2());
        return values_.block(start_row, start_col, rows, cols);
      }

      /*
      inline
      Eigen::Block<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >
      const_block(int start_row, int start_col, int rows, int cols) {
        assert(is_allocated());
        assert(start_row + rows <= size1());
        assert(start_col + cols <= size2());
        return values_.block(start_row, start_col, rows, cols);
      }
       */

      ALPS_STRONG_INLINE
      Eigen::Block<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >
      block() {
        assert(is_allocated());
        return values_.block(0, 0, size1_, size2_);
      }

      inline
      const Eigen::Block<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >
      block() const {
        assert(is_allocated());
        return values_.block(0, 0, size1_, size2_);
      }

      /*
      inline
      const Eigen::Block<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >
      const_block() {
        assert(is_allocated());
        return values_.block(0, 0, size1_, size2_);
      }
      */

    private:
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

      int size1_, size2_; //current size of ResizableMatrix
      eigen_matrix_t values_;
    };

    //some functions for compatibility
    template<typename Scalar>
    ALPS_STRONG_INLINE
    int num_cols(const alps::fastupdate::ResizableMatrix<Scalar> &m) {
      return m.size1();
    }

    template<typename Scalar>
    ALPS_STRONG_INLINE
    int num_rows(const alps::fastupdate::ResizableMatrix<Scalar> &m) {
      return m.size2();
    }

    template<typename Scalar>
    ALPS_STRONG_INLINE
    void gemm(const alps::fastupdate::ResizableMatrix<Scalar> &a, const alps::fastupdate::ResizableMatrix<Scalar> &b,
              alps::fastupdate::ResizableMatrix<Scalar> &c) {
      c = a * b;
    }

    template<typename Scalar>
    ALPS_STRONG_INLINE
    alps::fastupdate::ResizableMatrix<Scalar> inverse(const alps::fastupdate::ResizableMatrix<Scalar> &m) {
      alps::fastupdate::ResizableMatrix<Scalar> inv_m(m);
      inv_m.invert();
      return inv_m;
    }

    template<typename Scalar>
    ALPS_STRONG_INLINE
    Scalar determinant(const alps::fastupdate::ResizableMatrix<Scalar> &m) {
      return m.determinant();
    }

    template<typename Scalar>
    ALPS_STRONG_INLINE
    Scalar phase_of_determinant(const alps::fastupdate::ResizableMatrix<Scalar> &m) {
      Eigen::FullPivLU<Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > lu(m.block());
      Scalar phase = 1.0;
      for (int i=0; i<m.size1(); ++i) {
        Scalar val = lu.matrixLU()(i,i);
        if (val != 0.0) {
          phase *= val/std::abs(val);
        } else {
          phase = 0.0;
        }
      }
      return phase * static_cast<Scalar>(lu.permutationP().determinant() * lu.permutationQ().determinant());
    }

    //template<typename Scalar>
    //ALPS_STRONG_INLINE
    //Scalar safe_determinant(const alps::fastupdate::ResizableMatrix<Scalar> &m) {
      //return m.safe_determinant();
    //}


    template<typename Scalar>
    ALPS_STRONG_INLINE
    double norm_square(const alps::fastupdate::ResizableMatrix<Scalar> &m) {
      return m.block().squaredNorm();
    }

    //template<typename Scalar, typename Derived>
    //ALPS_STRONG_INLINE
    //Scalar determinant(const Eigen::MatrixBase<Derived>& m) {
    ////return m.determinant();
    //}

    template<typename M1, typename M2>
    ALPS_STRONG_INLINE
    void copy_block(const M1 &src, int start_row_src, int start_col_src,
                    M2 &dst, int start_row_dst, int start_col_dst,
                    int num_rows_block, int num_cols_block) {
      dst.block(start_row_dst, start_col_dst, num_rows_block, num_cols_block)
        = src.block(start_row_src, start_col_src, num_rows_block, num_cols_block);
    }

    template<typename Scalar>
    std::ostream &operator<<(std::ostream &os, const ResizableMatrix<Scalar> &m) {
      os << "memory size1: " << m.size1() << std::endl;
      os << "memory size2: " << m.size2() << std::endl;
      os << m.block() << std::endl;
      return os;
    }
  }
}

namespace std {
  template<typename Scalar>
  void swap(alps::fastupdate::ResizableMatrix<Scalar>& m1, alps::fastupdate::ResizableMatrix<Scalar>& m2) throw ()
  {
    m1.swap(m2);
  }
}


