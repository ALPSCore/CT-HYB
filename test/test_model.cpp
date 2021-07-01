#include <tuple>
#include <alps/params.hpp>

#include "../src/model/model.hpp"

#include <gtest.h>


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


// three-orbital t2g model without bath
/*
TEST(ModelLibrary, t2gModel) {
  typedef std::complex<double> SCALAR;
  SCALAR tval = 0.0;
  auto ntau = 1000;
  auto nflavors = 6;
  auto onsite_U = 2.0;
  auto JH = 0.1;

  alps::params par;
  par["model.sites"] = 3;
  par["model.spins"] = 2;
  par["model.n_tau_hyb"] = ntau;
  par["model.onsite_U"] = onsite_U;
  par["model.beta"] = 1.0;

  std::vector<std::tuple<int, int, int, int, SCALAR> > Uval_list{ {0, 1, 1, 0, -onsite_U} };
  std::vector<std::tuple<int, int, SCALAR> > t_list;
  boost::multi_array<SCALAR, 3> F(boost::extents[nflavors][nflavors][ntau+1]);
  std::fill(F.origin(), F.origin() + F.num_elements(), 0.0);

  ImpurityModelEigenBasis<SCALAR>::define_parameters(par);
  ImpurityModelEigenBasis<SCALAR> model(par, t_list, Uval_list, F);

  ASSERT_EQ(model.num_sectors(), 4);
  std::vector<int> min_enes_ref = {0, 2, 2, 2};
  std::vector<int> min_enes;
  for (auto sector=0; sector<4; ++sector) {
      min_enes.push_back(model.min_energy(sector));
  }
  ASSERT_EQ(min_enes, min_enes_ref);
}
*/