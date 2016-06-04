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
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace alps {
  template<typename T1, typename T2>
  T1 mypow(T1 x, T2 N) {
    return std::pow(x,N);
  }

  /*
  template<typename T2>
  boost::multiprecision::cpp_dec_float_50 mypow(boost::multiprecision::cpp_dec_float_50 x, T2 N) {
    return boost::multiprecision::pow(x,N);
  }

  template<typename T2>
  std::complex<boost::multiprecision::cpp_dec_float_50> mypow(std::complex<boost::multiprecision::cpp_dec_float_50> x, T2 N) {
    return boost::multiprecision::pow(x,N);
  }
  */

  //Compute the determinant of a matrix avoiding underflow and overflow
  //Note: This make a copy of the matrix.
  template<typename ReturnType, typename Derived>
  ReturnType
  safe_determinant(const Eigen::MatrixBase<Derived> &mat) {
    typedef typename Derived::RealScalar RealScalar;
    assert(mat.rows() == mat.cols());
    const int N = mat.rows();
    if (N == 0) {
      return ReturnType(1.0);
    }
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> mat_copy(mat);
    const RealScalar max_coeff = mat_copy.cwiseAbs().maxCoeff();
    if (max_coeff == 0.0) {
      return ReturnType(0.0);
    }
    mat_copy /= max_coeff;
    return ReturnType(mat_copy.determinant()) * mypow(ReturnType(max_coeff), 1. * N);
  }

  //Compute the determinant of a matrix avoiding underflow and overflow
  //Note: This make a copy of the matrix.
  template<typename ReturnType, typename Derived>
  ReturnType
  safe_determinant_eigen_block(const Eigen::Block<const Derived>& mat) {
    typedef typename Derived::Scalar Scalar;
    typedef typename Derived::RealScalar RealScalar;

    assert(mat.rows()==mat.cols());
    const int N = mat.rows();
    if (N==0) {
      return 1.0;
    }
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> mat_copy(mat);
    const RealScalar max_coeff = mat_copy.cwiseAbs().maxCoeff();
    if (max_coeff==0.0) {
      return 0.0;
    }
    mat_copy /= max_coeff;
    return ReturnType(mat_copy.determinant())*mypow(max_coeff, 1.*N);
  }

//Compute the inverse of a matrix avoiding underflow and overflow
//Note: This make a copy of the matrix.
  template<typename Derived>
  inline
  void
  safe_invert_in_place(Eigen::MatrixBase<Derived> &mat) {
    typedef typename Derived::RealScalar RealScalar;

    const int N = mat.rows();
    const RealScalar max_coeff = mat.cwiseAbs().maxCoeff();

    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> mat_copy = mat / max_coeff;
    mat = mat_copy.inverse() / max_coeff;
  }
}

namespace alps {

  template<typename Scalar>
  class ResizableMatrix {
  public:
    typedef Scalar type;

    ResizableMatrix(int size1, int size2) :
      size1_(size1),
      size2_(size2),
      values_(size1, size2)
    {
      assert(size1>=0 && size2>=0);
    }

    ResizableMatrix() : size1_(-1), size2_(-1), values_(0,0) {}

    ResizableMatrix(const ResizableMatrix<Scalar> &M) {
      if(M.is_allocated()) {
        size1_ = M.size1_;
        size2_ = M.size2_;
        values_ = M.values_;
      } else {
        size1_ = -1;
        size2_ = -1;
      }
    }

    inline bool is_allocated() const {
      return size1_>=0 && size2_>=0;
    }

    inline Scalar& operator()(const int i, const int j) {
      assert(is_allocated());
      assert(i<=size1_);
      assert(j<=size2_);
      return values_(i,j);
    }

    inline const Scalar& operator()(const int i, const int j) const {
      assert(is_allocated());
      assert(i<=size1_);
      assert(j<=size2_);
      return values_(i,j);
    }

    //ResizableMatrix size
    inline int size1() const {assert(is_allocated()); return size1_;}
    inline int size2() const {assert(is_allocated()); return size2_;}
    inline int memory_size1() const {assert(is_allocated()); return values_.rows();}
    inline int memory_size2() const {assert(is_allocated()); return values_.cols();}

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
        if (size1>memory_size1() || size2>memory_size2()) {
          values_.conservativeResize(1.2*size1+1, 1.2*size2+1);
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
        if (size1>memory_size1() || size2>memory_size2()) {
          values_.resize(1.2*size1+1, 1.2*size2+1);
        }
      }
      size1_ = size1;
      size2_ = size2;
    }

    //delete last row and column
    inline void remove_row_col_last() {
      assert(is_allocated());
      assert(size1_>0);
      assert(size2_>0);
      --size1_;
      --size2_;
    }

    //delete last row
    inline void remove_row_last() {
      assert(is_allocated());
      assert(size1_>0);
      --size1_;
    }

    //delete last column
    inline void remove_col_last() {
      assert(is_allocated());
      assert(size2_>0);
      --size2_;
    }

    //swap two rows
    inline void swap_row(int r1, int r2) {
      assert(is_allocated());
      assert(r1<size1_);
      assert(r2<size1_);
      values_.row(r1).swap(values_.row(r2));
    }

    //swap two columns
    inline void swap_col(int c1, int c2) {
      assert(is_allocated());
      assert(c1<size2_);
      assert(c2<size2_);
      values_.col(c1).swap(values_.col(c2));
    }

    //swap two rows and columns
    inline void swap_row_col(int c1, int c2) {
      swap_col(c1,c2);
      swap_row(c1,c2);
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
      assert(size1_==size2_);
      if (size1_>0) {
        block().setIdentity();
      }
    }

    //This is well-defined for a square matrix
    inline Scalar determinant() const {
      assert(is_allocated());
      assert(size1_==size2_);

      const int size = size1_;

      //the simple ones...
      if(size==0) return 1;
      if(size==1) return operator()(0,0);
      if(size==2) return operator()(0,0)*operator()(1,1)-operator()(0,1)*operator()(1,0);

      return block().determinant();
    }

    //This is well-defined for a square matrix
    template<typename ExtendedScalar>
    ExtendedScalar safe_determinant() const {
      assert(is_allocated());
      assert(size1_==size2_);

      const int size = size1_;

      //the simple ones...
      if(size==0) return 1;
      if(size==1) return operator()(0,0);
      if(size==2) return operator()(0,0)*operator()(1,1)-operator()(0,1)*operator()(1,0);

      return safe_determinant_eigen_block<ExtendedScalar>(block());
    }

    inline void clear() {
      if (is_allocated()) {
        block().setZero();
      }
    }

    inline ResizableMatrix operator-(const ResizableMatrix<Scalar> &M2) const {
      assert(is_allocated());
      assert(size1_==M2.size1_);
      assert(size2_==M2.size2_);

      ResizableMatrix Msum(*this);
      Msum.values_.block(0,0,size1_,size2_) -= M2.values_.block(0,0,size1_,size2_);
      return Msum;
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
      assert(size1_==size2_);
      eigen_matrix_t inv = block().inverse();
      values_ = inv;//Should we use std::swap(*values_,inv)?
    }

    inline void safe_invert() {
      assert(is_allocated());
      assert(size1_==size2_);
      eigen_matrix_t inv = block();
      safe_invert_in_place(inv);
      values_ = inv;//Should we use std::swap(*values_,inv)?
    }

    inline Scalar trace() {
      assert(is_allocated());
      assert(size1_ == size2_);
      return block().trace();
    }

    inline
    Eigen::Block<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >
    block(int start_row, int start_col, int rows, int cols) {
      assert(is_allocated());
      assert(start_row+rows<=size1());
      assert(start_col+cols<=size2());
      return values_.block(start_row, start_col, rows, cols);
    }

    inline
    Eigen::Block<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> >
    block(int start_row, int start_col, int rows, int cols) const {
      assert(is_allocated());
      assert(start_row+rows<=size1());
      assert(start_col+cols<=size2());
      return values_.block(start_row, start_col, rows, cols);
    }

    inline
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

    //Only for debug and test
    //Please do not access the raw pointer
    //Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>* const get_pointer() const {
      //return values_;
    //}

  private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_t;

    int size1_, size2_; //current size of ResizableMatrix
    eigen_matrix_t values_;
  };

  /*
  template<typename Scalar>
  std::ostream &operator<<(std::ostream &os, const ResizableMatrix<Scalar> &M){
    os<<"[ ";
    for(int i=0;i<M.size();++i){
      //os<<"[ ";
      for(int j=0;j<M.size();++j){
        os<<M(i,j)<<" ";
      }
      if(i<M.size()-1)
        os<<" ;"<<" ";
    }
    os<<"]"<<" ";
    return os;
  }

  template<typename Scalar>
  std::ostream   &operator<<(std::ostream  &os, const ResizableMatrix<Scalar> &M); //forward declaration
  */
}

namespace std {
  template<typename Scalar>
  void swap(alps::ResizableMatrix<Scalar>& m1, alps::ResizableMatrix<Scalar>& m2) throw ()
  {
    m1.swap(m2);
  }
}
