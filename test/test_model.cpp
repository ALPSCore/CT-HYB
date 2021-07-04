#include <tuple>
#include <alps/params.hpp>

# include "test_model.hpp"

// Single-orbital model without bath
TEST(ModelLibrary, SingleOrbitalModel) {
  typedef std::complex<double> SCALAR;
  SCALAR tval = 0.0;
  auto ntau = 1000;
  auto nflavors = 2;

  alps::params par;
  par["model.sites"] = 1;
  par["model.spins"] = 2;
  par["model.n_tau_hyb"] = ntau;
  double onsite_U = 2.0;
  par["model.onsite_U"] = onsite_U;
  par["model.beta"] = 1.0;

  std::vector<std::tuple<int, int, int, int, SCALAR> > Uval_list{ {0, 1, 1, 0, -onsite_U} };
  std::vector<std::tuple<int, int, SCALAR> > t_list;
  boost::multi_array<SCALAR, 3> F(boost::extents[nflavors][nflavors][ntau+1]);
  std::fill(F.origin(), F.origin() + F.num_elements(), 0.0);

  ImpurityModelEigenBasis<SCALAR>::define_parameters(par);
  ImpurityModelEigenBasis<SCALAR> model(par, t_list, Uval_list, F);

  check_eigenes(model);

  ASSERT_EQ(model.num_sectors(), 4);
  std::vector<int> min_enes_ref = {0, 2, 2, 2};
  std::vector<int> nelec_sectors_ref = {2, 1, 1, 0};
  std::vector<int> min_enes, nelec_sectors;
  for (auto sector=0; sector<4; ++sector) {
      min_enes.push_back(model.min_energy(sector));
      nelec_sectors.push_back(model.nelec_sector(sector));
  }
  ASSERT_EQ(min_enes, min_enes_ref);
  ASSERT_EQ(nelec_sectors, nelec_sectors_ref);
}


// three-orbital t2g model with SK interactions
TEST(ModelLibrary, t2gModel) {
  typedef std::complex<double> SCALAR;
  SCALAR tval = 0.0;
  auto ntau = 1000;
  auto nflavors = 6;
  auto norbs = nflavors / 2;
  auto onsite_U = 2.0;
  auto JH = 0.1;
  auto calU = onsite_U - 3*JH;
  auto mu_half = 2.5*JH; // A. Gergoes (2013) seems to use this value. (Why?)

  alps::params par;
  par["model.sites"] = 3;
  par["model.spins"] = 2;
  par["model.n_tau_hyb"] = ntau;
  par["model.onsite_U"] = onsite_U;
  par["model.beta"] = 1.0;

  std::vector<std::tuple<int, int, SCALAR> > t_list;
  for (auto f=0; f<2*norbs; ++f) {
      t_list.push_back({f, f, -mu_half});
  }
  auto Uval_list = create_SK_Uijkl<SCALAR>(onsite_U, JH);

  boost::multi_array<SCALAR, 3> F(boost::extents[nflavors][nflavors][ntau+1]);
  std::fill(F.origin(), F.origin() + F.num_elements(), 0.0);

  ImpurityModelEigenBasis<SCALAR>::define_parameters(par);
  ImpurityModelEigenBasis<SCALAR> model(par, t_list, Uval_list, F);

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