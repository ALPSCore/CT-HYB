#include "impurity.hpp"

inline void print_acc_rate(const alps::accumulators::result_set &results, const std::string &name, std::ostream &os) {
  os << " " << name + " : "
     << results[name + "_accepted_scalar"].mean<double>()
         / results[name + "_valid_move_scalar"].mean<double>()
     << std::endl;
}

template<typename T>
struct to_complex {
  std::complex<T> operator()(const T &re, const T &im) {
    return std::complex<T>(re, im);
  }
};

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::show_statistics(const alps::accumulators::result_set &results) {
#ifdef MEASURE_TIMING
  const std::vector<double> timings = results["TimingsSecPerNMEAS"].template mean<std::vector<double> >();
  std::cout << std::endl << "==== Timings analysis ====" << std::endl;
  std::cout << " The following are the timings per window sweep (in units of second): " << std::endl;
  std::cout << " Local updates (insertion/removal/shift of operators/worm: " << timings[0] << std::endl;
  std::cout << " Global updates (global shift etc.): " << timings[1] << std::endl;
  std::cout << " Worm measurement: " << timings[2] << std::endl;
  std::cout << " Non worm measurement: " << timings[3] << std::endl;
#endif

  std::cout << std::endl << "==== Thermalization analysis ====" << std::endl;
  std::cout << boost::format("Perturbation orders just before and after measurement steps are %1% and %2%.") %
      results["Pert_order_start"].template mean<double>() %
      results["Pert_order_end"].template mean<double>() << std::endl;

  std::cout << std::endl << "==== Number of Monte Carlo steps spent in configuration spaces ====" << std::endl;
  std::cout << "Z function" << " : " << results["Z_function_space_num_steps"].template mean<double>() << std::endl;
  for (int w = 0; w < worm_types.size(); ++w) {
    std::cout << get_config_space_name(worm_types[w]) << " : "
              << results["worm_space_num_steps_" + get_config_space_name(worm_types[w])].template mean<double>()
              << std::endl;
  }

  std::cout << std::endl << "==== Acceptance updates of operators hybridized with bath ====" << std::endl;
  for (int k = 1; k < par["update.multi_pair_ins_rem"].template as<int>() + 1; ++k) {
    print_acc_rate(results, ins_rem_updater[k-1]->get_name(), std::cout);
  }
  print_acc_rate(results, single_op_shift_updater.get_name(), std::cout);
  print_acc_rate(results, operator_pair_flavor_updater.get_name(), std::cout);

  std::cout << std::endl << "==== Acceptance rates of worm updates ====" << std::endl;
  std::vector<std::string> active_worm_updaters = get_active_worm_updaters();
  for (int iu = 0; iu < active_worm_updaters.size(); ++iu) {
    print_acc_rate(results, active_worm_updaters[iu], std::cout);
  }
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::compute_G1(
    const alps::accumulators::result_set &results,
    std::map<std::string,boost::any> &ar) {
  namespace g=alps::gf;
  typedef Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
  typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> complex_matrix_t;

  const int n_tau(par["measurement.G1.n_tau"]);
  const int n_matsubara(par["measurement.G1.n_matsubara"]);
  const double beta(par["model.beta"]);
  const int n_flavors = par["model.sites"].template as<int>() * par["model.spins"].template as<int>();
  const double temperature(1.0 / beta);
  const double sign = results["Sign"].template mean<double>();
  //The factor of temperature below comes from the extra degree of freedom for beta in the worm
  const double coeff =
      results["worm_space_volume_G1"].template mean<double>() /
          (sign * results["Z_function_space_volume"].template mean<double>());


  boost::shared_ptr<OrthogonalBasis> p_basis = p_G1_meas->get_p_basis_f();
  const int dim_ir_basis = p_basis->dim();

  boost::multi_array<std::complex<double>, 3>
      Gl_org_basis(boost::extents[n_flavors][n_flavors][dim_ir_basis]);
  {
    const std::vector<double> Gl_Re = results["G1_Re"].template mean<std::vector<double> >();
    const std::vector<double> Gl_Im = results["G1_Im"].template mean<std::vector<double> >();
    assert(Gl_Re.size() == n_flavors * n_flavors * dim_ir_basis);
    boost::multi_array<std::complex<double>, 3>
        Gl(boost::extents[n_flavors][n_flavors][dim_ir_basis]);
    std::transform(Gl_Re.begin(), Gl_Re.end(), Gl_Im.begin(), Gl.origin(), to_complex<double>());
    std::transform(Gl.origin(), Gl.origin() + Gl.num_elements(), Gl.origin(),
                   std::bind1st(std::multiplies<std::complex<double> >(), coeff));

    //rotate back to the original basis
    complex_matrix_t mattmp(n_flavors, n_flavors), mattmp2(n_flavors, n_flavors);
    const matrix_t inv_rotmat_Delta = p_model->get_rotmat_Delta().inverse();
    for (int il = 0; il < dim_ir_basis; ++il) {
      for (int flavor1 = 0; flavor1 < n_flavors; ++flavor1) {
        for (int flavor2 = 0; flavor2 < n_flavors; ++flavor2) {
          mattmp(flavor1, flavor2) = Gl[flavor1][flavor2][il];
        }
      }
      mattmp2 = (p_model->get_rotmat_Delta()) * mattmp * inv_rotmat_Delta;
      for (int flavor1 = 0; flavor1 < n_flavors; ++flavor1) {
        for (int flavor2 = 0; flavor2 < n_flavors; ++flavor2) {
          Gl_org_basis[flavor1][flavor2][il] = mattmp2(flavor1, flavor2);
        }
      }
    }
  }
  ar["G1_IR"] = Gl_org_basis;

  /*
   * Compute G(tau) from Legendre coefficients
   */
  typedef alps::gf::three_index_gf<std::complex<double>, alps::gf::itime_mesh,
                                   alps::gf::index_mesh,
                                   alps::gf::index_mesh
  > ITIME_GF;

  ITIME_GF
      itime_gf(alps::gf::itime_mesh(beta, n_tau + 1), alps::gf::index_mesh(n_flavors), alps::gf::index_mesh(n_flavors));
  std::vector<double> Pvals(dim_ir_basis);
  std::vector<double> sqrt_array(dim_ir_basis);
  for (int il = 0; il < dim_ir_basis; ++il) {
    sqrt_array[il] = sqrt(2.0/p_basis->norm2(il));
  }

  for (int itau = 0; itau < n_tau + 1; ++itau) {
    const double tau = itau * (beta / n_tau);
    const double x = 2 * tau / beta - 1.0;
    p_basis->value(x, Pvals); //Compute P_l[x]

    for (int flavor = 0; flavor < n_flavors; ++flavor) {
      for (int flavor2 = 0; flavor2 < n_flavors; ++flavor2) {
        for (int il = 0; il < dim_ir_basis; ++il) {
          itime_gf(g::itime_index(itau), g::index(flavor), g::index(flavor2)) +=
              Pvals[il] * Gl_org_basis[flavor][flavor2][il] * sqrt_array[il] * temperature;
        }
      }
    }
  }
  ar["gtau"] = itime_gf;

  /*
   * Compute Gomega from Legendre coefficients
   */
  typedef alps::gf::three_index_gf<std::complex<double>, alps::gf::matsubara_positive_mesh,
                                   alps::gf::index_mesh,
                                   alps::gf::index_mesh
  > GOMEGA;

  const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &Tnl(p_basis->Tnl(n_matsubara));
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> tmp_mat(dim_ir_basis, 1), tmp_mat2(n_matsubara, 1);
  GOMEGA gomega(alps::gf::matsubara_positive_mesh(beta, n_matsubara),
                alps::gf::index_mesh(n_flavors),
                alps::gf::index_mesh(n_flavors));
  for (int flavor = 0; flavor < n_flavors; ++flavor) {
    for (int flavor2 = 0; flavor2 < n_flavors; ++flavor2) {
      for (int il = 0; il < dim_ir_basis; ++il) {
        tmp_mat(il, 0) = Gl_org_basis[flavor][flavor2][il];
      }
      tmp_mat2 = Tnl * tmp_mat;
      for (int im = 0; im < n_matsubara; ++im) {
        gomega(g::matsubara_index(im), g::index(flavor), g::index(flavor2)) = tmp_mat2(im, 0);
      }
    }
  }
  ar["gf"] = gomega;
}

//very crapy way to implement ...
template<typename SCALAR>
void rotate_back_G2(int n_flavors, boost::multi_array<std::complex<double>, 4> &G2,
                    const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic> &rotmat_Delta) {

  boost::multi_array<std::complex<double>, 4>
      work(boost::extents[n_flavors][n_flavors][n_flavors][n_flavors]);

  //transform the first index
  std::fill(work.origin(), work.origin() + work.num_elements(), 0.0);
  for (int f1 = 0; f1 < n_flavors; ++f1) {
    for (int f2 = 0; f2 < n_flavors; ++f2) {
      for (int f3 = 0; f3 < n_flavors; ++f3) {
        for (int f4 = 0; f4 < n_flavors; ++f4) {
          for (int g1 = 0; g1 < n_flavors; ++g1) {
            work[f1][f2][f3][f4] += rotmat_Delta(f1, g1) * G2[g1][f2][f3][f4];
          }
        }
      }
    }
  }
  G2 = work;

  //transform the second index
  std::fill(work.origin(), work.origin() + work.num_elements(), 0.0);
  for (int f1 = 0; f1 < n_flavors; ++f1) {
    for (int f2 = 0; f2 < n_flavors; ++f2) {
      for (int f3 = 0; f3 < n_flavors; ++f3) {
        for (int f4 = 0; f4 < n_flavors; ++f4) {
          for (int g2 = 0; g2 < n_flavors; ++g2) {
            work[f1][f2][f3][f4] += myconj(rotmat_Delta(f2, g2)) * G2[f1][g2][f3][f4];
          }
        }
      }
    }
  }
  G2 = work;

  //transform the third index
  std::fill(work.origin(), work.origin() + work.num_elements(), 0.0);
  for (int f1 = 0; f1 < n_flavors; ++f1) {
    for (int f2 = 0; f2 < n_flavors; ++f2) {
      for (int f3 = 0; f3 < n_flavors; ++f3) {
        for (int f4 = 0; f4 < n_flavors; ++f4) {
          for (int g3 = 0; g3 < n_flavors; ++g3) {
            work[f1][f2][f3][f4] += rotmat_Delta(f3, g3) * G2[f1][f2][g3][f4];
          }
        }
      }
    }
  }
  G2 = work;

  //transform the fourth index
  std::fill(work.origin(), work.origin() + work.num_elements(), 0.0);
  for (int f1 = 0; f1 < n_flavors; ++f1) {
    for (int f2 = 0; f2 < n_flavors; ++f2) {
      for (int f3 = 0; f3 < n_flavors; ++f3) {
        for (int f4 = 0; f4 < n_flavors; ++f4) {
          for (int g4 = 0; g4 < n_flavors; ++g4) {
            work[f1][f2][f3][f4] += myconj(rotmat_Delta(f4, g4)) * G2[f1][f2][f3][g4];
          }
        }
      }
    }
  }
  G2 = work;
}

template<typename IMP_MODEL>
void HybridizationSimulation<IMP_MODEL>::compute_G2(
    const alps::accumulators::result_set &results,
    std::map<std::string,boost::any> &ar) {
  const int n_flavors = par["model.sites"].template as<int>() * par["model.spins"].template as<int>();
  const double sign = results["Sign"].template mean<double>();

  //The factor of temperature below comes from the extra degree of freedom for beta in the worm
  const double coeff =
      results["worm_space_volume_G2"].template mean<double>() /
          (sign * results["Z_function_space_volume"].template mean<double>());

  boost::shared_ptr<OrthogonalBasis> p_basis_f = p_G2_meas->get_p_basis_f();
  boost::shared_ptr<OrthogonalBasis> p_basis_b = p_G2_meas->get_p_basis_b();

  const int dim_f = p_basis_f->dim();
  const int dim_b = p_basis_b->dim();

  const std::vector<double> Gl_Re = results["G2_Re"].template mean<std::vector<double> >();
  const std::vector<double> Gl_Im = results["G2_Im"].template mean<std::vector<double> >();
  boost::multi_array<std::complex<double>, 7>
      Gl(boost::extents[n_flavors][n_flavors][n_flavors][n_flavors][dim_f][dim_f][dim_b]);
  std::transform(Gl_Re.begin(), Gl_Re.end(), Gl_Im.begin(), Gl.origin(), to_complex<double>());
  std::transform(Gl.origin(), Gl.origin() + Gl.num_elements(), Gl.origin(),
                 std::bind1st(std::multiplies<std::complex<double> >(), coeff));

  //rotate back to the original basis (using not-cache-friendly approach...)
  for (int il = 0; il < dim_f; ++il) {
    for (int il2 = 0; il2 < dim_f; ++il2) {
      for (int il3 = 0; il3 < dim_b; ++il3) {

        //copy data to work1
        boost::multi_array<std::complex<double>, 4>
            work(boost::extents[n_flavors][n_flavors][n_flavors][n_flavors]);
        for (int f1 = 0; f1 < n_flavors; ++f1) {
          for (int f2 = 0; f2 < n_flavors; ++f2) {
            for (int f3 = 0; f3 < n_flavors; ++f3) {
              for (int f4 = 0; f4 < n_flavors; ++f4) {
                work[f1][f2][f3][f4] = Gl[f1][f2][f3][f4][il][il2][il3];
              }
            }
          }
        }

        rotate_back_G2(n_flavors, work, p_model->get_rotmat_Delta());

        //copy result to Gl
        for (int f1 = 0; f1 < n_flavors; ++f1) {
          for (int f2 = 0; f2 < n_flavors; ++f2) {
            for (int f3 = 0; f3 < n_flavors; ++f3) {
              for (int f4 = 0; f4 < n_flavors; ++f4) {
                Gl[f1][f2][f3][f4][il][il2][il3] = work[f1][f2][f3][f4];
              }
            }
          }
        }

      }
    }
  }

  ar["G2_LEGENDRE"] = Gl;
}
