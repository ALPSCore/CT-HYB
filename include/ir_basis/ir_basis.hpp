#pragma once

#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <assert.h>

#include <boost/multi_array.hpp>
#include <boost/type_traits.hpp>
//#include <boost/typeof/typeof.hpp>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

//#include <boost/timer/timer.hpp>

#include <alps/gf/mesh.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>

extern "C" void dgesvd_(const char *jobu, const char *jobvt,
                        const int *m, const int *n, double *a, const int *lda,
                        double *s, double *u, const int *ldu,
                        double *vt, const int *ldvt,
                        double *work, const int *lwork, int *info);


namespace ir {
  enum statistics { fermionic, bosonic };

  namespace detail {
    /*
    template<class T, int k>
    void compute_transformation_matrix_to_matsubara(
        int n,
        statistics s,
        const std::vector<alps::gf::piecewise_polynomial<T, k> > &bf_src,
        std::vector<std::complex<double> > &Tnl
    );
     */
  }

  template<typename T>
  // we expect T = double
  alps::gf::piecewise_polynomial<T, 3> construct_piecewise_polynomial_cspline(
      const std::vector<double> &x_array, const std::vector<double> &y_array) {

    const int n_points = x_array.size();
    const int n_section = n_points - 1;

    boost::multi_array<double, 2> coeff(boost::extents[n_section][4]);

    gsl_interp_accel *my_accel_ptr = gsl_interp_accel_alloc();
    gsl_spline *my_spline_ptr = gsl_spline_alloc(gsl_interp_cspline, n_points);
    gsl_spline_init(my_spline_ptr, &x_array[0], &y_array[0], n_points);

    // perform spline interpolation
    for (int s = 0; s < n_section; ++s) {
      const double dx = x_array[s + 1] - x_array[s];
      coeff[s][0] = y_array[s];
      coeff[s][1] = gsl_spline_eval_deriv(my_spline_ptr, x_array[s], my_accel_ptr);
      coeff[s][2] = 0.5 * gsl_spline_eval_deriv2(my_spline_ptr, x_array[s], my_accel_ptr);
      coeff[s][3] =
          (y_array[s + 1] - y_array[s] - coeff[s][1] * dx - coeff[s][2] * dx * dx) / (dx * dx * dx);//ugly hack
      assert(
          std::abs(
              y_array[s + 1] - y_array[s] - coeff[s][1] * dx - coeff[s][2] * dx * dx - coeff[s][3] * dx * dx * dx)
              < 1e-8
      );
    }

    gsl_spline_free(my_spline_ptr);
    gsl_interp_accel_free(my_accel_ptr);

    return alps::gf::piecewise_polynomial<T, 3>(n_section, x_array, coeff);
  };


  /**
   * Fermionic kernel
   */
  class FermionicKernel {
   public:
    FermionicKernel(double Lambda) : Lambda_(Lambda) {}

    double operator()(double x, double y) const {
      const double limit = 100.0;
      if (Lambda_ * y > limit) {
        return std::exp(-0.5 * Lambda_ * x * y - 0.5 * Lambda_ * y);
      } else if (Lambda_ * y < -limit) {
        return std::exp(-0.5 * Lambda_ * x * y + 0.5 * Lambda_ * y);
      } else {
        return std::exp(-0.5 * Lambda_ * x * y) / (2 * std::cosh(0.5 * Lambda_ * y));
      }
    }

    static statistics get_statistics() {
      return fermionic;
    }

   private:
    double Lambda_;
  };

  /**
   * Bosonic kernel
   */
  class BosonicKernel {
   public:
    BosonicKernel(double Lambda) : Lambda_(Lambda) {}

    double operator()(double x, double y) const {
      const double limit = 100.0;
      if (std::abs(Lambda_ * y) < 1e-10) {
        return std::exp(-0.5 * Lambda_ * x * y) / Lambda_;
      } else if (Lambda_ * y > limit) {
        return y * std::exp(-0.5 * Lambda_ * x * y - 0.5 * Lambda_ * y);
      } else if (Lambda_ * y < -limit) {
        return -y * std::exp(-0.5 * Lambda_ * x * y + 0.5 * Lambda_ * y);
      } else {
        return y * std::exp(-0.5 * Lambda_ * x * y) / (2 * std::sinh(0.5 * Lambda_ * y));
      }
    }

    static statistics get_statistics() {
      return bosonic;
    }

   private:
    double Lambda_;
  };

/**
 * Class template for kernel Ir basis
 */
  template<typename Scalar, typename Kernel>
  class Basis {
   public:
    Basis(double Lambda, int max_dim, double cutoff = 1e-10, int N = 501);

   private:
    typedef alps::gf::piecewise_polynomial<double, 3> pp_type;

    std::vector<pp_type> basis_functions_;

   public:
    void value(double x, std::vector<double> &val) const;
    const pp_type &operator()(int l) const { return basis_functions_[l]; }
    int dim() const { return basis_functions_.size(); }

    void compute_Tnl(
        int n_min, int n_max,
        boost::multi_array<std::complex<double>, 2> &Tnl
    ) const;
  };

  /**
   * Typedefs for convenience
   */
  typedef Basis<double, FermionicKernel> FermionicBasis;
  typedef Basis<double, BosonicKernel> BosonicBasis;
}

#include "ir_basis.ipp"
