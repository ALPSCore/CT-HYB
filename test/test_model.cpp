#include <tuple>
#include <alps/params.hpp>

#include "../src/model/model.hpp"

#include <gtest.h>

// Check all eigen energies by applying H to an eigenstate.
template<typename SCALAR>
void
check_eigenes(const ImpurityModelEigenBasis<SCALAR> &model) {
  using dense_matrix_t = Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>;
  using braket_t = typename ImpurityModelEigenBasis<SCALAR>::BRAKET_T;

  for (auto sector=0; sector<model.num_sectors(); ++sector) {
    int dim_sector = model.dim_sector(sector);
    for (auto state=0; state<dim_sector; ++state) {
      dense_matrix_t eigenbasis(dim_sector, 1);
      eigenbasis.setZero();
      eigenbasis(state, 0) = 1.0; // sector-th eigenbasis
      auto eigenket = braket_t(sector, eigenbasis);

      auto ene = 0.0;

      // hopping term
      for (const auto& tij: model.get_nonzero_t_vals()) {
        auto ket = braket_t(eigenket);
        model.apply_op_hyb_ket(ANNIHILATION_OP, std::get<1>(tij), ket);
        model.apply_op_hyb_ket(CREATION_OP, std::get<0>(tij), ket);
        ene += get_real(std::get<2>(tij) * model.product(eigenket, ket));
      }

      // Coulomb interaction
      for (const auto& Uijkl: model.get_nonzero_U_vals()) {
        auto ket = braket_t(eigenket);
        model.apply_op_hyb_ket(ANNIHILATION_OP, std::get<3>(Uijkl), ket);
        model.apply_op_hyb_ket(ANNIHILATION_OP, std::get<2>(Uijkl), ket);
        model.apply_op_hyb_ket(CREATION_OP,     std::get<1>(Uijkl), ket);
        model.apply_op_hyb_ket(CREATION_OP,     std::get<0>(Uijkl), ket);
        ene += get_real(std::get<4>(Uijkl) * model.product(eigenket, ket));
      }

      ASSERT_NEAR(ene, model.get_eigenvals_sector()[sector][state] + model.get_reference_energy(), 1e-8);
    }
  }

}


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