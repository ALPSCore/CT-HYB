#include "ir_test.hpp"

TEST(PiecewisePolynomial, Orthogonalization) {
    typedef double Scalar;
    const int n_section = 10, k = 8, n_basis = 3;
    typedef alps::gf::piecewise_polynomial<Scalar,k> pp_type;

    std::vector<double> section_edges(n_section+1);
    boost::multi_array<Scalar,3> coeff(boost::extents[n_basis][n_section][k+1]);

    for (int s = 0; s < n_section + 1; ++s) {
        section_edges[s] = s*2.0/n_section - 1.0;
    }
    section_edges[0] = -1.0;
    section_edges[n_section] = 1.0;

    std::vector<pp_type> nfunctions;

    // x^0, x^1, x^2, ...
    for (int n = 0; n < n_basis; ++ n) {
        boost::multi_array<Scalar,2> coeff(boost::extents[n_section][k+1]);
        std::fill(coeff.origin(), coeff.origin()+coeff.num_elements(), 0.0);

        for (int s = 0; s < n_section; ++s) {
            double rtmp = 1.0;
            for (int l = 0; l < k + 1; ++l) {
                if (n - l < 0) {
                    break;
                }
                if (l > 0) {
                    rtmp /= l;
                    rtmp *= n + 1 - l;
                }
                coeff[s][l] = rtmp * std::pow(section_edges[s], n-l);
            }
        }

        nfunctions.push_back(pp_type(n_section, section_edges, coeff));
    }

    // Check if correctly constructed
    double x = 0.9;
    for (int n = 0; n < n_basis; ++ n) {
        EXPECT_NEAR(nfunctions[n].compute_value(x), std::pow(x, n), 1e-8);
    }

    // Check overlap
    for (int n = 0; n < n_basis; ++ n) {
        for (int m = 0; m < n_basis; ++ m) {
            EXPECT_NEAR(nfunctions[n].overlap(nfunctions[m]), (std::pow(1.0,n+m+1)-std::pow(-1.0,n+m+1))/(n+m+1), 1e-8);
        }
    }


    // Check plus and minus
    for (int n = 0; n < n_basis; ++ n) {
        EXPECT_NEAR(4 * nfunctions[n].compute_value(x), (4.0*nfunctions[n]).compute_value(x), 1e-8);
        for (int m = 0; m < n_basis; ++m) {
            EXPECT_NEAR(nfunctions[n].compute_value(x) + nfunctions[m].compute_value(x),
                        (nfunctions[n] + nfunctions[m]).compute_value(x), 1e-8);
            EXPECT_NEAR(nfunctions[n].compute_value(x) - nfunctions[m].compute_value(x),
                        (nfunctions[n] - nfunctions[m]).compute_value(x), 1e-8);
        }
    }

    alps::gf::orthonormalize(nfunctions);
    for (int n = 0; n < n_basis; ++ n) {
        for (int m = 0; m < n_basis; ++m) {
            EXPECT_NEAR(nfunctions[n].overlap(nfunctions[m]),
                        n == m ? 1.0 : 0.0,
                        1e-8
            );
        }
    }

    //l = 0 should be x
    EXPECT_NEAR(nfunctions[1].compute_value(x) * std::sqrt(2.0/3.0), x, 1E-8);
}

template<class T>
class HighTTest : public testing::Test {
};

typedef ::testing::Types<ir::FermionicKernel, ir::BosonicKernel> KernelTypes;

TYPED_TEST_CASE(HighTTest, KernelTypes);

TYPED_TEST(HighTTest, KernelTypes) {
  try {
    //construct ir basis
    const double Lambda = 0.01;//high T
    const int max_dim = 100;
    ir::Basis<double,TypeParam> basis(Lambda, max_dim);
    ASSERT_TRUE(basis.dim()>3);

    //IR basis functions should match Legendre polynomials
    const int N = 10;
    for (int i = 1; i < N - 1; ++i) {
      const double x = i * (2.0/(N-1)) - 1.0;

      double rtmp;

      //l = 0
      rtmp = basis(0).compute_value(x);
      ASSERT_TRUE(std::abs(rtmp-std::sqrt(0+0.5)) < 0.02);

      //l = 1
      rtmp = basis(1).compute_value(x);
      ASSERT_TRUE(std::abs(rtmp-std::sqrt(1+0.5)*x) < 0.02);

      //l = 2
      rtmp = basis(2).compute_value(x);
      ASSERT_TRUE(std::abs(rtmp-std::sqrt(2+0.5)*(1.5*x*x-0.5)) < 0.02);
    }

    //check parity
    {
      double sign = -1.0;
      double x = 1.0;
      for (int l = 0; l < basis.dim(); ++l) {
        ASSERT_NEAR(basis(l).compute_value(x) + sign * basis(l).compute_value(-x), 0.0, 1e-8);
        sign *= -1;
      }
    }

    //check transformation matrix to Matsubara frequencies
    if (TypeParam::get_statistics() == ir::fermionic) {

      const int N_iw = 3;
      //time_t t1, t2;

      boost::multi_array<std::complex<double>,2> Tnl_legendre(boost::extents[N_iw][3]),
          Tnl_ir(boost::extents[N_iw][3]);

      compute_Tnl_legendre(N_iw, 3, Tnl_legendre);

      //Fast version
      basis.compute_Tnl(0, N_iw-1, Tnl_ir);
      for (int n = 0; n < N_iw; n++) {
        for (int l = 0; l < 3; ++l) {
          ASSERT_NEAR(std::abs(Tnl_ir[n][l] / (Tnl_legendre[n][l]) - 1.0), 0.0, 1e-5);
        }
      }
    }
  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}

TEST(IrBasis, FermionInsulatingGtau) {
  try {
    const double Lambda = 300.0, beta = 100.0;
    const int max_dim = 100;
    ir::Basis<double,ir::FermionicKernel> basis(Lambda, max_dim, 1e-10, 501);
    ASSERT_TRUE(basis.dim()>0);

    typedef alps::gf::piecewise_polynomial<double,3> pp_type;

    const int nptr = basis(0).num_sections() + 1;
    std::vector<double> x(nptr), y(nptr);
    for (int i = 0; i < nptr; ++i) {
      x[i] = basis(0).section_edge(i);
      y[i] = std::exp(-0.5*beta)*std::cosh(-0.5*beta*x[i]);
    }
    pp_type gtau(ir::construct_piecewise_polynomial_cspline<double>(x, y));

    std::vector<double> coeff(basis.dim());
    for (int l = 0; l < basis.dim(); ++l) {
      coeff[l] = gtau.overlap(basis(l)) * beta / std::sqrt(2.0);
    }

    std::vector<double> y_r(nptr, 0.0);
    for (int l = 0; l < 30; ++l) {
      for (int i = 0; i < nptr; ++i) {
        y_r[i] += coeff[l] * (std::sqrt(2.0)/beta) * basis(l).compute_value(x[i]);
      }
    }

    double max_diff = 0.0;
    for (int i = 0; i < nptr; ++i) {
      max_diff = std::max(std::abs(y[i]-y_r[i]), max_diff);
      ASSERT_TRUE(std::abs(y[i]-y_r[i]) < 1e-6);
    }

    //to matsubara freq.
    const int n_iw = 1000;
    boost::multi_array<std::complex<double>,2> Tnl(boost::extents[n_iw][basis.dim()]);
    basis.compute_Tnl(0, n_iw-1, Tnl);
    Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic> Tnl_mat(n_iw, basis.dim()), coeff_vec(basis.dim(),1);
    coeff_vec.setZero();
    for (int l = 0; l < basis.dim(); ++l) {
      coeff_vec(l,0) = coeff[l];
      for (int n = 0; n < n_iw; ++n) {
        Tnl_mat(n,l) = Tnl[n][l];
      }
    }
    Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic> coeff_iw = Tnl_mat * coeff_vec;

    const std::complex<double> zi(0.0, 1.0);
    for (int n = 0; n < n_iw; ++n) {
      double wn = (2.*n+1)*M_PI/beta;
      std::complex<double> z = - 0.5/(zi*wn - 1.0) - 0.5/(zi*wn + 1.0);
      ASSERT_NEAR(std::abs(z-coeff_iw(n)), 0.0, 1e-8);
    }


  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
}
