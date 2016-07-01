#pragma once

#include <algorithm>
#include <iterator>

#include <Eigen/Core>


namespace alps {
  namespace fastupdate {
    namespace detail {
      //return permutation from time-ordering (1 or -1)
      template <typename InputIterator>
      int permutation(InputIterator begin, InputIterator end) {
        using std::swap;
        typedef typename std::iterator_traits<InputIterator>::value_type my_value_type;

        std::vector<my_value_type> values;
        std::copy(begin, end, std::back_inserter(values));

        const int N = values.size();
        int perm = 1;
        while (true) {
          bool exchanged = false;
          for (int i=0; i<N-1; ++i) {
            if ( !(values[i]<values[i+1]) ) {
              swap(values[i], values[i+1]);
              perm *= -1;
              exchanged = true;
            }
          }
          if (!exchanged) break;
        }
        return perm;
      }

      template<typename Derived>
      inline typename Derived::RealScalar max_abs_coeff(const Eigen::MatrixBase<Derived>& mat) {
        typedef typename Derived::RealScalar RealScalar;
        const int rows = mat.rows();
        const int cols = mat.cols();

        RealScalar result = 0.0;
        for (int j=0; j<cols; ++j) {
          for (int i=0; i<rows; ++i) {
            result = std::max(result, std::abs(mat(i,j)));
          }
        }

        return result;
      }

      //Compute the determinant of a matrix avoiding underflow and overflow
      //Note: This make a copy of the matrix.
      template<typename Derived>
      typename Derived::Scalar
      safe_determinant(const Eigen::MatrixBase<Derived>& mat) {
        typedef typename Derived::RealScalar RealScalar;
        assert(mat.rows()==mat.cols());
        const int N = mat.rows();
        if (N==0) {
          return 1.0;
        }
        Eigen::Matrix<typename Derived::Scalar,Eigen::Dynamic,Eigen::Dynamic> mat_copy(mat);
        const RealScalar max_coeff = mat_copy.cwiseAbs().maxCoeff();
        if (max_coeff==0.0) {
          return 0.0;
        }
        mat_copy /= max_coeff;
        return mat_copy.determinant()*std::pow(max_coeff, 1.*N);
      }

      //Compute the determinant of a matrix avoiding underflow and overflow
      //Note: This make a copy of the matrix.
      template<typename Derived>
      typename Derived::Scalar
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
        return mat_copy.determinant()*std::pow(max_coeff, 1.*N);
      }

      //Compute the inverse of a matrix avoiding underflow and overflow
      //Note: This make a copy of the matrix.
      template<typename Derived>
      inline
      void
      safe_invert_in_place(Eigen::MatrixBase<Derived>& mat) {
        typedef typename Derived::RealScalar RealScalar;

        const int N = mat.rows();
        const RealScalar max_coeff = mat.cwiseAbs().maxCoeff();

        Eigen::Matrix<typename Derived::Scalar,Eigen::Dynamic,Eigen::Dynamic> mat_copy = mat/max_coeff;
        mat = mat_copy.inverse()/max_coeff;
      }

      //Compute the inverse of a matrix avoiding underflow and overflow
      //Note: This make a copy of the matrix.
      template<typename Derived>
      Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
      safe_inverse(const Eigen::MatrixBase<Derived>& mat) {
        typedef typename Derived::RealScalar RealScalar;

        const int N = mat.rows();
        const RealScalar max_coeff = mat.cwiseAbs().maxCoeff();

        Eigen::Matrix<typename Derived::Scalar,Eigen::Dynamic,Eigen::Dynamic> mat_copy = mat/max_coeff;
        return mat_copy.inverse()/max_coeff;
      }

      //Compute the inverse of a matrix avoiding underflow and overflow
      //Note: This make a copy of the matrix.
      template<typename Derived>
      Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
      safe_inverse(const Eigen::Block<const Derived>& mat) {
        typedef typename Derived::RealScalar RealScalar;

        const int N = mat.rows();
        const RealScalar max_coeff = mat.cwiseAbs().maxCoeff();

        Eigen::Matrix<typename Derived::Scalar,Eigen::Dynamic,Eigen::Dynamic> mat_copy = mat/max_coeff;
        return mat_copy.inverse()/max_coeff;
      }

      //Compute the inverse of a matrix avoiding underflow and overflow
      //Note: This make a copy of the matrix.
      template<typename Derived>
      Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
      safe_inverse(Eigen::Block<Derived>& mat) {
        typedef typename Derived::RealScalar RealScalar;

        const int N = mat.rows();
        const RealScalar max_coeff = mat.cwiseAbs().maxCoeff();

        Eigen::Matrix<typename Derived::Scalar,Eigen::Dynamic,Eigen::Dynamic> mat_copy = mat/max_coeff;
        return mat_copy.inverse()/max_coeff;
      }

      //Comb sort (in ascending order). Returns the permutation sign (1 or -1)
      //Assumption: no duplicate members
      //Compare: lesser operator
      template<typename Iterator, typename Compare>
      int
      comb_sort(Iterator first, Iterator end, const Compare& compare) {
        using std::swap;

        const int N = std::distance(first, end);

        int count_exchange = 0;

        int gap = static_cast<int>(N/1.3);
        while(true) {
          bool swapped = false;

          Iterator it1 = first;
          Iterator it2 = first;
          std::advance(it2, gap);
          for (int i=0; i+gap<N; ++i) {
            if (compare(*it2, *it1)) {
              swap(*it1, *it2);
              ++count_exchange;
              swapped = true;
            }
            ++it1;
            ++it2;
          }

          if (!swapped && gap==1) {
            break;
          }

          gap = std::max(static_cast<int>(gap/1.3),1);
        }

        return count_exchange%2==0 ? 1 : -1;
      }

      inline bool my_isnan(double x) {
        return std::isnan(x);
      }

      inline bool my_isnan(std::complex<double> x) {
        return my_isnan(x.real()) || my_isnan(x.imag());
      }

      template<typename Scalar, typename M>
      std::vector<Scalar>
      lu_product(const M& matrix) {
        Eigen::FullPivLU<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> > lu(matrix);
        const int size1 = lu.rows();
        std::vector<Scalar> results(size1);
        for (int i = 0; i < size1; ++i) {
          results[i] = lu.matrixLU()(i,i);
        }
        results[0] *= lu.permutationP().determinant()*lu.permutationQ().determinant();
        return results;
      };

      template<typename Scalar>
      bool lesser_by_abs(const Scalar& v1, const Scalar& v2) {
        return std::abs(v1) < std::abs(v2);
      }
    }
  }
}
