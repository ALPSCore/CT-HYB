#include <tuple>
#include <alps/params.hpp>

# include "test_model.hpp"

// Single-orbital model without bath
TEST(ModelLibrary, SingleOrbitalModel) {
  typedef std::complex<double> SCALAR;
  SCALAR tval = 0.0;
  auto nflavors = 2;

  double onsite_U = 2.0;

  // Repulsive onsite U
  std::vector<std::tuple<int, int, int, int, SCALAR> > Uval_list{ {0, 1, 1, 0, onsite_U} };
  std::vector<std::tuple<int, int, SCALAR> > t_list;

  AtomicModelEigenBasis<SCALAR> model(nflavors, t_list, Uval_list);

  check_eigenes(model);

  ASSERT_EQ(model.num_sectors(), 4);
  std::vector<int> min_enes_ref = {2, 0, 0, 0};
  std::vector<int> nelec_sectors_ref = {2, 1, 1, 0};
  std::vector<int> min_enes, nelec_sectors;
  for (auto sector=0; sector<4; ++sector) {
      min_enes.push_back(model.min_energy(sector) + model.get_reference_energy());
      nelec_sectors.push_back(model.nelec_sector(sector));
  }
  ASSERT_EQ(min_enes_ref, min_enes);
  ASSERT_EQ(nelec_sectors_ref, nelec_sectors);

  model.save_info_for_postprocessing("single_orb_model.h5");
}


// three-orbital t2g model with SK interactions
TEST(ModelLibrary, t2gModel) {
  typedef std::complex<double> SCALAR;
  SCALAR tval = 0.0;
  auto nflavors = 6;
  auto norbs = nflavors / 2;
  auto onsite_U = 2.0;
  auto JH = 0.1;
  auto calU = onsite_U - 3*JH;
  auto mu_half = 2.5*JH; // A. Gergoes (2013) seems to use this value. (Why?)

  alps::params par;
  par["model.sites"] = 3;
  par["model.spins"] = 2;

  std::vector<std::tuple<int, int, SCALAR> > t_list;
  for (auto f=0; f<2*norbs; ++f) {
      t_list.push_back({f, f, -mu_half});
  }
  auto Uval_list = create_SK_Uijkl<SCALAR>(onsite_U, JH);
  //debug
  //Uval_list.resize(0);

  AtomicModelEigenBasis<SCALAR>::define_parameters(par);
  AtomicModelEigenBasis<SCALAR> model(nflavors, t_list, Uval_list);

  check_eigenes(model);
  check_sector_propagate(model);

  // Check the lowest eigen energy for each nelec sector
  // Table I of A. Georges et al., Annu Rev Conden Ma P 4, 137–178 (2013).
  std::vector<double> lowest_enes_ref{0.0, -2.5*JH, calU-5*JH, 3*calU-7.5*JH, 6*calU - 5*JH, 10*calU - 2.5*JH, 15*calU};
  std::vector<double> lowest_enes(nflavors+1, 1E+100);
  for (auto sector=0; sector<model.num_sectors(); ++sector) {
      int nelec = model.nelec_sector(sector);
      lowest_enes[nelec] = std::min(lowest_enes[nelec], model.min_energy(sector) + model.get_reference_energy());
  }
  for (auto n=0; n<nflavors+1; ++n) {
      ASSERT_NEAR(lowest_enes[n], lowest_enes_ref[n], 1e-8);
  }

  // Check apply exp(- tau H)
}

TEST(Hyb, two_flavor) {
  test_read_intpl_hyb_two_flavor<double>();
  test_read_intpl_hyb_two_flavor<std::complex<double>>();
}

/*
TEST(Model, ops_at_same_time) {
  using SCALAR = std::complex<double>;
  using MODEL = COMPLEX_EIGEN_BASIS_MODEL;

  auto beta = 10.0;
  auto tau = 0.1*beta;

  auto up = 0;
  auto dn = 1;

  operator_container_t ops;
  auto op1 = psi(OperatorTime(tau, 0), CREATION_OP,     up);
  auto op2 = psi(OperatorTime(tau, 0), ANNIHILATION_OP, up);
  ops.insert(op1);
  ops.insert(op2);
  ASSERT_EQ(ops.size(), 2);
  ASSERT_EQ(ops.erase(op1), 1);
  ASSERT_EQ(ops.erase(op2), 1);
}
*/

TEST(Model, op_with_time_deriv) {
  auto t1 = 0.0;
  auto flavor = 0;
  auto op1 = psi(OperatorTime(t1), CREATION_OP, flavor);
  auto op2 = psi(OperatorTime(t1), CREATION_OP, flavor, true);
  ASSERT_TRUE(op1 != op2);
}