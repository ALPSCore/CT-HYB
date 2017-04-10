#include "ir_basis.hpp"

namespace ir {
  typedef alps::gf::piecewise_polynomial<std::complex<double>, 4> piecewise_polynomial_dcomplex_4;
  typedef alps::gf::piecewise_polynomial<double, 3> piecewise_polynomial_double_3;

  namespace detail {
    //AVOID USING BOOST_TYPEOF
    template <class T1, class T2>
    struct result_of_multiply {
      typedef std::complex<double> value;
    };

    template <>
    struct result_of_multiply<double,double> {
      typedef double value;
    };

    template<typename T>
    inline std::vector<T> linspace(T minval, T maxval, int N) {
      std::vector<T> r(N);
      for (int i = 0; i < N; ++i) {
        r[i] = i * (maxval - minval) / (N - 1) + minval;
      }
      return r;
    }

    template<class Matrix, class Vector>
    void svd_square_matrix(Matrix &K, int n, Vector &S, Matrix &Vt, Matrix &U) {
      char jobu = 'S';
      char jobvt = 'S';
      int lda = n;
      int ldu = n;
      int ldvt = n;

      double *vt = Vt.data();
      double *u = U.data();
      double *s = S.data();

      double dummywork;
      int lwork = -1;
      int info = 0;

      double *A = K.data();

      //get optimal workspace
      dgesvd_(&jobu, &jobvt, &n, &n, A, &lda, s, u, &ldu, vt, &ldvt, &dummywork, &lwork, &info);

      lwork = int(dummywork) + 32;
      Vector work(lwork);

      dgesvd_(&jobu, &jobvt, &n, &n, A, &lda, s, u, &ldu, vt, &ldvt, &work[0], &lwork, &info);
      if (info != 0) {
        throw std::runtime_error("SVD failed to converge!");
      }
    }

    template<class T, int k>
    void construct_matsubra_basis_functions(
        int n_min, int n_max,
        statistics s,
        const std::vector<double> &section_edges,
        std::vector<alps::gf::piecewise_polynomial<std::complex<T>, k> > &results) {
      typedef alps::gf::piecewise_polynomial<std::complex<T>, k> pp_type;

      const int N = section_edges.size() - 1;

      results.resize(0);

      std::complex<double> z;
      boost::multi_array<std::complex<T>, 2> coeffs(boost::extents[N][k + 1]);

      boost::array<double,k+1> pre_factor;
      pre_factor[0] = 1.0;
      for (int j = 1; j < k+1; ++j) {
        pre_factor[j] = pre_factor[j-1]/j;
      }

      for (int n = n_min; n <= n_max; ++n) {
        if (s == fermionic) {
          z = -std::complex<double>(0.0, n + 0.5) * M_PI;
        } else if (s == bosonic) {
          z = -std::complex<double>(0.0, n) * M_PI;
        }
        for (int section = 0; section < N; ++section) {
          const double x = section_edges[section];
          std::complex<T> exp0 = std::exp(z * (x + 1));
          std::complex<T> z_power = 1.0;
          for (int j = 0; j < k+1; ++j) {
            coeffs[section][j] = exp0 * z_power * pre_factor[j];
            z_power *= z;
          }
        }
        results.push_back(pp_type(N, section_edges, coeffs));
      }
    }

    template<class T, int k, int k_iw>
    void compute_transformation_matrix_to_matsubara(
        int n_min, int n_max,
        statistics statis,
        const std::vector<alps::gf::piecewise_polynomial<T,k> >& bf_src,
        boost::multi_array<std::complex<double>,2> & Tnl
    ) {
      typedef std::complex<double> dcomplex;
      typedef alps::gf::piecewise_polynomial<std::complex<double>, k_iw> pp_type;
      typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

      std::vector<pp_type> matsubara_functions;

      construct_matsubra_basis_functions(n_min, n_max, statis, bf_src[0].section_edges(), matsubara_functions);

      const int n_section = bf_src[0].num_sections();
      const int n_iw = n_max-n_min+1;

      matrix_t left_mid_matrix(n_iw, k+1);
      matrix_t left_matrix(n_iw, k_iw+1);
      matrix_t mid_matrix(k_iw+1, k+1);
      matrix_t right_matrix(k+1, bf_src.size());
      matrix_t r(n_iw, bf_src.size());
      r.setZero();

      boost::array<double, k + k_iw + 2> dx_power;

      const double cutoff = 0.1;
      for (int s=0; s < n_section; ++s) {
        double x0 = bf_src[0].section_edge(s);
        double x1 = bf_src[0].section_edge(s+1);
        double dx = x1 - x0;

        dx_power[0] = 1.0;
        for (int p = 1; p < dx_power.size(); ++p) {
          dx_power[p] = dx * dx_power[p - 1];
        }

        //Use Taylor expansion for exp(i w_n tau) for M_PI*(n+0.5)*dx < cutoff*M_PI
        int n_max_cs = std::max(std::min(static_cast<int>(cutoff/dx - 0.5), n_max), 0);

        for (int p = 0; p < k_iw + 1; ++p) {
          for (int p2 = 0; p2 < k + 1; ++p2) {
            mid_matrix(p, p2) = dx_power[p + p2 + 1] / (p + p2 + 1.0);
          }
        }

        for (int n = 0; n < n_max_cs-n_min+1; ++n) {
          for (int p = 0; p < k_iw + 1; ++p) {
            left_matrix(n, p) = alps::gf::detail::conjg(matsubara_functions[n].coefficient(s, p));
          }
        }

        left_mid_matrix.block(0,0,n_max_cs-n_min+1,k+1) = left_matrix.block(0,0,n_max_cs-n_min+1,k_iw+1) * mid_matrix;

        //Compute the overlap exactly for M_PI*(n+0.5)*dx > cutoff*M_PI
        for (int n = n_max_cs+1; n <= n_max; ++n) {
          std::complex<double> z;
          if (statis == fermionic) {
            z = std::complex<double>(0.0, n+0.5) * M_PI;
          } else if (statis == bosonic) {
            z = std::complex<double>(0.0, n) * M_PI;
          }

          dcomplex dx_z = dx * z;
          dcomplex dx_z2 = dx_z * dx_z;
          dcomplex dx_z3 = dx_z2 * dx_z;
          dcomplex inv_z = 1.0/z;
          dcomplex inv_z2 = inv_z * inv_z;
          dcomplex inv_z3 = inv_z2 * inv_z;
          dcomplex inv_z4 = inv_z3 * inv_z;
          dcomplex exp = std::exp(dx * z);
          dcomplex exp0 = std::exp((x0+1.0) * z);

          left_mid_matrix(n-n_min,0) = (-1.0+exp)*inv_z*exp0;
          left_mid_matrix(n-n_min,1) = ((dx_z -1.0)*exp+1.0)*inv_z2*exp0;
          left_mid_matrix(n-n_min,2) = ((dx_z2-2.0*dx_z+2.0)*exp-2.0)*inv_z3*exp0;
          left_mid_matrix(n-n_min,3) = ((dx_z3-3.0*dx_z2+6.0*dx_z-6.0)*exp+6.0)*inv_z4*exp0;
        }

        for (int l = 0; l < bf_src.size(); ++l) {
          for (int p2 = 0; p2 < k + 1; ++p2) {
            right_matrix(p2, l) = bf_src[l].coefficient(s, p2);
          }
        }

        r += left_mid_matrix * right_matrix;
      }

      Tnl.resize(boost::extents[n_iw][bf_src.size()]);
      std::vector<double> inv_norm(bf_src.size());
      for (int l=0; l<bf_src.size(); ++l) {
        inv_norm[l] = 1./std::sqrt(static_cast<double>(bf_src[l].overlap(bf_src[l])));
      }
      for (int n=0; n<n_iw; ++n) {
        for (int l=0; l<bf_src.size(); ++l) {
          // 0.5 is the inverse of the norm of exp(i w_n tau)
          Tnl[n][l] = r(n,l) * inv_norm[l] * std::sqrt(0.5);
        }
      }
    }

  }//namespace detail

  template<typename Kernel>
  void do_svd(double Lambda, int parity, int N, double cutoff_singular_values,
              std::vector<double> &singular_values,
              std::vector<piecewise_polynomial_double_3> &basis_functions
  ) {
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

    double de_cutoff = 2.5;

    //DE mesh for x
    std::vector<double> tx_vec = detail::linspace<double>(0.0, de_cutoff, N);
    std::vector<double> weight_x(N), x_vec(N);
    for (int i = 0; i < N; ++i) {
      x_vec[i] = std::tanh(0.5 * M_PI * std::sinh(tx_vec[i]));
      //sqrt of the weight of DE formula
      weight_x[i] = std::sqrt(0.5 * M_PI * std::cosh(tx_vec[i])) / std::cosh(0.5 * M_PI * std::sinh(tx_vec[i]));
    }

    //DE mesh for y
    std::vector<double> ty_vec = detail::linspace<double>(-de_cutoff, 0.0, N);
    std::vector<double> y_vec(N), weight_y(N);
    for (int i = 0; i < N; ++i) {
      y_vec[i] = std::tanh(0.5 * M_PI * std::sinh(ty_vec[i])) + 1.0;
      //sqrt of the weight of DE formula
      weight_y[i] = std::sqrt(0.5 * M_PI * std::cosh(ty_vec[i])) / std::cosh(0.5 * M_PI * std::sinh(ty_vec[i]));
    }

    Kernel k_obj(Lambda);
    matrix_t K(N, N);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        K(i, j) = weight_x[i] * (k_obj(x_vec[i], y_vec[j]) + parity * k_obj(x_vec[i], -y_vec[j])) * weight_y[j];
      }
    }

    //Perform SVD
    Eigen::VectorXd svalues(N);
    matrix_t U(N, N), Vt(N, N);
    detail::svd_square_matrix(K, N, svalues, Vt, U);

    //Count non-zero SV
    int dim = N;
    for (int i = 1; i < N; ++i) {
      if (std::abs(svalues(i) / svalues(0)) < cutoff_singular_values) {
        dim = i;
        break;
      }
    }

    //Rescale U and V
    U.conservativeResize(N, dim);
    for (int l = 0; l < dim; ++l) {
      for (int i = 0; i < N; ++i) {
        U(i, l) /= weight_x[i];
      }
      if (U(N - 1, l) < 0.0) {
        U.col(l) *= -1;
      }
    }

    singular_values.resize(dim);
    for (int l = 0; l < dim; ++l) {
      singular_values[l] = svalues(l);
    }

    //cubic spline interpolation
    const int n_points = 2 * N + 1;
    const int n_section = n_points + 1;
    std::vector<double> x_array(n_points), y_array(n_points);

    //set up x values
    for (int itau = 0; itau < N; ++itau) {
      x_array[-itau + n_points / 2] = -x_vec[itau];
      x_array[itau + n_points / 2] = x_vec[itau];
    }
    x_array.front() = -1.0;
    x_array.back() = 1.0;

    // spline interpolation
    for (int l = 0; l < dim; ++l) {
      //set up y values
      for (int itau = 0; itau < N; ++itau) {
        y_array[-itau + n_points / 2] = parity * U(itau, l);
        y_array[itau + n_points / 2] = U(itau, l);
      }
      if (parity == -1) {
        y_array[n_points / 2] = 0.0;
      }
      y_array.front() = parity * U(N - 1, l);
      y_array.back() = U(N - 1, l);

      basis_functions.push_back(construct_piecewise_polynomial_cspline<double>(x_array, y_array));
    }

    orthonormalize(basis_functions);
    assert(singular_values.size() == basis_functions.size());
  }

  template<typename Scalar, typename Kernel>
  Basis<Scalar, Kernel>::Basis(double Lambda, int max_dim, double cutoff, int N) {

    std::vector<double> even_svalues, odd_svalues, svalues;
    std::vector<pp_type> even_basis_functions, odd_basis_functions;

    do_svd<Kernel>(Lambda, 1, N, cutoff, even_svalues, even_basis_functions);
    do_svd<Kernel>(Lambda, -1, N, cutoff, odd_svalues, odd_basis_functions);

    //Merge
    basis_functions_.resize(0);
    assert(even_basis_functions.size() == even_svalues.size());
    assert(odd_basis_functions.size() == odd_svalues.size());
    for (int pair = 0; pair < std::max(even_svalues.size(), odd_svalues.size()); ++pair) {
      if (pair < even_svalues.size()) {
        svalues.push_back(even_svalues[pair]);
        basis_functions_.push_back(even_basis_functions[pair]);
      }
      if (pair < odd_svalues.size()) {
        svalues.push_back(odd_svalues[pair]);
        basis_functions_.push_back(odd_basis_functions[pair]);
      }
    }

    assert(even_svalues.size() + odd_svalues.size() == svalues.size());

    //use max_dim
    if (svalues.size() > max_dim) {
      svalues.resize(max_dim);
      basis_functions_.resize(max_dim);
    }

    //Check
    for (int i = 0; i < svalues.size() - 1; ++i) {
      if (svalues[i] < svalues[i + 1]) {
        //FIXME: SHOULD NOT THROW IN A CONSTRUCTOR
        throw std::runtime_error("Even and odd basis functions do not appear alternately.");
      }
    }
  };

  template<typename Scalar, typename Kernel>
  void
  Basis<Scalar, Kernel>::value(double x, std::vector<double> &val) const {
    assert(val.size() >= basis_functions_.size());
    assert(x >= -1.00001 && x <= 1.00001);

    const int dim = basis_functions_.size();

    if (dim > val.size()) {
      val.resize(dim);
    }
    const int section = basis_functions_[0].find_section(x);
    for (int l = 0; l < dim; l++) {
      val[l] = basis_functions_[l].compute_value(x, section);
    }
  }

  template<typename Scalar, typename Kernel>
  void
  Basis<Scalar, Kernel>::compute_Tnl(
      int n_min, int n_max,
      boost::multi_array<std::complex<double>, 2> &Tnl
  ) const {
    detail::compute_transformation_matrix_to_matsubara<double,3,16>(n_min,
                                                                  n_max,
                                                                  Kernel::get_statistics(),
                                                                  basis_functions_,
                                                                  Tnl);
  };

  /// Compute overlap <left | right> with complex conjugate
  template<class T1, int k1, class T2, int k2>
  void compute_overlap(
      const std::vector<alps::gf::piecewise_polynomial<T1, k1> > &left_vectors,
      const std::vector<alps::gf::piecewise_polynomial<T2, k2> > &right_vectors,
      boost::multi_array<typename detail::result_of_multiply<T1,T2>::value, 2> &results) {
    typedef typename detail::result_of_multiply<T1,T2>::value Tr;

    const int NL = left_vectors.size();
    const int NR = right_vectors.size();
    const int n_sections = left_vectors[0].num_sections();

    if (left_vectors[0].section_edges() != right_vectors[0].section_edges()) {
      throw std::runtime_error("Not supported");
    }

    for (int n = 0; n < NL - 1; ++n) {
      if (left_vectors[n].section_edges() != left_vectors[n + 1].section_edges()) {
        throw std::runtime_error("Not supported");
      }
    }

    for (int l = 0; l < NR - 1; ++l) {
      if (right_vectors[l].section_edges() != right_vectors[l + 1].section_edges()) {
        throw std::runtime_error("Not supported");
      }
    }

    boost::array<double, k1 + k2 + 2> x_min_power, dx_power;

    Eigen::Matrix<Tr, k1 + 1, k2 + 1> mid_matrix;
    Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> left_matrix(NL, k1 + 1);
    Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic> right_matrix(k2 + 1, NR);
    Eigen::Matrix<Tr, Eigen::Dynamic, Eigen::Dynamic> r(NL, NR);

    r.setZero();
    for (int s = 0; s < n_sections; ++s) {
      //boost::timer::cpu_timer t1;
      dx_power[0] = 1.0;
      const double dx = left_vectors[0].section_edge(s + 1) - left_vectors[0].section_edge(s);
      for (int p = 1; p < dx_power.size(); ++p) {
        dx_power[p] = dx * dx_power[p - 1];
      }

      for (int p = 0; p < k1 + 1; ++p) {
        for (int p2 = 0; p2 < k2 + 1; ++p2) {
          mid_matrix(p, p2) = dx_power[p + p2 + 1] / (p + p2 + 1.0);
        }
      }

      for (int n = 0; n < NL; ++n) {
        for (int p = 0; p < k1 + 1; ++p) {
          //FIXME: DO NOT USE alps::gf::detail::conjg
          left_matrix(n, p) = alps::gf::detail::conjg(left_vectors[n].coefficient(s, p));
        }
      }

      for (int l = 0; l < NR; ++l) {
        for (int p2 = 0; p2 < k2 + 1; ++p2) {
          right_matrix(p2, l) = right_vectors[l].coefficient(s, p2);
        }
      }
      //boost::timer::cpu_timer t2;

      r += left_matrix * (mid_matrix * right_matrix);
      //boost::timer::cpu_timer t3;

    }

    results.resize(boost::extents[NL][NR]);
    for (int n = 0; n < NL; ++n) {
      for (int l = 0; l < NR; ++l) {
        results[n][l] = r(n, l);
      }
    }

  }
}
