#pragma once

#include <boost/multi_array.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

#include <iostream>
#include <math.h>
#include <vector>
#include <complex>

template<typename T> T mycast(std::complex<double> val);
template<typename T> T myconj(T val);
template<typename T> T mysign(T x);

inline double get_real(std::complex<double> x) {
    return x.real();
}

inline double get_imag(std::complex<double> x) {
    return x.imag();
}

inline double get_real(double x) {
    return x;
}

inline double get_imag(double x) {
    return 0.0;
}

template<typename T>
T mysign(T x) {
    return x/std::abs(x);
}

template<typename T>
bool my_equal(T x, T y, double eps=1E-8) {
    return std::abs(x-y)/std::max(std::abs(x),std::abs(y))<eps;
}

template<typename T>
bool my_rdiff(T x, T y) {
    return std::abs(x-y)/std::max(std::abs(x),std::abs(y));
}

template <typename Derived>
inline int size1(const Eigen::EigenBase<Derived>& mat) {
    return mat.rows();
}

template <typename Derived>
inline int size2(const Eigen::EigenBase<Derived>& mat) {
    return mat.cols();
}

template <typename Derived1, typename Derived2, typename Derived3>
inline void matrix_right_multiply(Eigen::EigenBase<Derived1>& mat1, Eigen::EigenBase<Derived2>& mat2, Eigen::EigenBase<Derived3>& res) {
    res = mat1*mat2;
}

template<typename T>
double maxAbsCoeff(const Eigen::SparseMatrix<T>& mat) {
    double maxval;
    int count = 0;
    for (int k=0; k<mat.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(mat,k); it; ++it) {
            maxval = count==0 ? std::abs(it.value()) : std::max(std::abs(it.value()), maxval);
        }
    }
    return maxval;
}

inline double min_distance(double dist, double BETA)
{
    const double abs_dist = std::abs(dist);
    assert(abs_dist>=0 && abs_dist<=BETA);
    return std::min(abs_dist, BETA-abs_dist);
}


template<typename SCALAR, typename M>
double spectral_norm_SVD(const M& mat) {
    typedef Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
    const double cutoff = 1E-15;
    const int rows = size1(mat);
    const int cols = size2(mat);
    if (rows==0 || cols==0) {
        return 0.0;
    }
    matrix_t mat_tmp(rows, cols);
    double max_abs = -1;
    for (int j=0; j<cols; ++j) {
        for (int i=0; i<rows; ++i) {
            mat_tmp(i,j) = mat(i,j);
            max_abs = std::max(max_abs,std::abs(mat_tmp(i,j)));
        }
    }
    if (max_abs==0.0) {
        return 0.0;
    }
    const double coeff = 1.0/max_abs;
    for (int j=0; j<cols; ++j) {
        for (int i=0; i<rows; ++i) {
            mat_tmp(i,j) *= coeff;
            if (std::abs(mat_tmp(i,j))<cutoff)  {
                mat_tmp(i,j) = 0.0;
            }
        }
    }
    const double tmp = mat_tmp.squaredNorm();
    Eigen::JacobiSVD<matrix_t> svd(mat_tmp);
#ifndef NDEBUG
    const int size_SVD = svd.singularValues().size();
    for (int i=0; i<size_SVD-1; ++i) {
        assert(std::abs(svd.singularValues()[i])>=std::abs(svd.singularValues()[i+1]));
    }
    if(isnan(std::abs(svd.singularValues()[0])/coeff)) {
        std::cout << "Norm is Nan" << std::endl;
        std::cout << "max_abs is " << max_abs << std::endl;
        std::cout << "coeff is " << coeff << std::endl;
        std::cout << "mat is " << std::endl << mat << std::endl;
        std::cout << "mat_tmp is " << std::endl << mat_tmp << std::endl;
        exit(-1);
    }
#endif
    const double norm = std::abs(svd.singularValues()[0])/coeff;
    assert(!isnan(norm));
    return norm;
}

template<typename SCALAR, typename M>
double spectral_norm_diag(const M& mat) {
    typedef Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
    const double cutoff = 1E-15;
    const int rows = size1(mat);
    const int cols = size2(mat);
    if (rows==0 || cols==0) {
        return 0.0;
    }
    matrix_t mat_tmp(rows, cols);
    double max_abs = -1;
    for (int j=0; j<cols; ++j) {
        for (int i=0; i<rows; ++i) {
            mat_tmp(i,j) = mat(i,j);
            max_abs = std::max(max_abs,std::abs(mat_tmp(i,j)));
        }
    }
    if (max_abs==0.0) {
        return 0.0;
    }
    const double coeff = 1.0/max_abs;
    for (int j=0; j<cols; ++j) {
        for (int i=0; i<rows; ++i) {
            mat_tmp(i,j) *= coeff;
            if (std::abs(mat_tmp(i,j))<cutoff)  {
                mat_tmp(i,j) = 0.0;
            }
        }
    }
    if (mat_tmp.rows()>mat_tmp.cols()) {
      mat_tmp = mat_tmp.adjoint()*mat_tmp;
    } else {
      mat_tmp = mat_tmp*mat_tmp.adjoint();
    }
    Eigen::SelfAdjointEigenSolver<matrix_t> esolv(mat_tmp,false);
    //Eigen::Matrix<double,Eigen::Dynamic,1> abs_evals = esolv.eigenvalues().cwiseAbs();
    const double norm = std::sqrt(esolv.eigenvalues().cwiseAbs().maxCoeff())/coeff;
    assert(!isnan(norm));
    return norm;
}

//Extract real parts of boost::muliti_array
template<class SCALAR,int DIMENSION>
boost::multi_array<double,DIMENSION>
get_real_parts(const boost::multi_array<SCALAR,DIMENSION>& data) {
    boost::multi_array<double,DIMENSION> real_part(data.shape());
    std::transform(data.begin(), data.end(), real_part.begin(), get_real);
    return real_part;
};

//Extract imaginary parts of boost::muliti_array
template<class SCALAR,int DIMENSION>
boost::multi_array<double,DIMENSION>
get_imag_parts(const boost::multi_array<SCALAR,DIMENSION>& data) {
    boost::multi_array<double,DIMENSION> imag_part(data.shape());
    std::transform(data.begin(), data.end(), imag_part.begin(), get_imag);
    return imag_part;
}

template <class RNG> double open_random(RNG& rng, double t1, double t2, double eps=1e-10) {return (t1-t2)*((1-eps)*rng()+eps/2)+t2;}
