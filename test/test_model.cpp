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
      dense_matrix_t evec(dim_sector, 1);
      evec.setZero();
      evec(state, 0) = 1.0; // sector-th eigenbasis
      auto eigenket = braket_t(sector, evec);
      auto eigenbra = braket_t(sector, evec.transpose());

      auto ene = 0.0;

      // hopping term
      for (const auto& tij: model.get_nonzero_t_vals()) {
        auto ket = braket_t(eigenket);
        model.apply_op_hyb_ket(ANNIHILATION_OP, std::get<1>(tij), ket);
        model.apply_op_hyb_ket(CREATION_OP, std::get<0>(tij), ket);
        ene += get_real(std::get<2>(tij) * model.product(eigenbra, ket));
      }

      // Coulomb interaction
      for (const auto& Uijkl: model.get_nonzero_U_vals()) {
        auto ket = braket_t(eigenket);
        model.apply_op_hyb_ket(ANNIHILATION_OP, std::get<3>(Uijkl), ket);
        model.apply_op_hyb_ket(ANNIHILATION_OP, std::get<2>(Uijkl), ket);
        model.apply_op_hyb_ket(CREATION_OP,     std::get<1>(Uijkl), ket);
        model.apply_op_hyb_ket(CREATION_OP,     std::get<0>(Uijkl), ket);
        ene += get_real(std::get<4>(Uijkl) * model.product(eigenbra, ket));
      }

      ASSERT_NEAR(ene, model.get_eigenvals_sector()[sector][state] + model.get_reference_energy(), 1e-8);
    }
  }

}

// Create U tensor for SK interaction for a t2g model
template<typename SCALAR>
std::vector<std::tuple<int,int,int,int,SCALAR>>
create_SK_Uijkl(double onsite_U, double JH) {
  std::vector<std::tuple<int, int, int, int, SCALAR> > Uval_list;
  auto norbs = 3;

  for (auto s1=0; s1<2; ++s1) {
    for (auto s2=0; s2<2; ++s2) {

      //U_{aaaa}
      for (auto a = 0; a < norbs; ++a) {
        Uval_list.push_back({2*a+s1, 2*a+s2, 2*a+s2, 2*a+s1, 0.5*onsite_U});
      }

      //U_{abba}
      for (auto a = 0; a < norbs; ++a) {
        for (auto b = 0; b < norbs; ++b) {
          if (a == b) {
            continue;
          }
          Uval_list.push_back({2*a+s1, 2*b+s2, 2*b+s2, 2*a+s1, 0.5*(onsite_U-2*JH)});
        }
      }

      //U_{abab} & U_{aabb}
      for (auto a = 0; a < norbs; ++a) {
        for (auto b = 0; b < norbs; ++b) {
          if (a == b) {
            continue;
          }
          Uval_list.push_back({2*a+s1, 2*b+s2, 2*a+s2, 2*b+s1, 0.5*JH});
          Uval_list.push_back({2*a+s1, 2*a+s2, 2*b+s2, 2*b+s1, 0.5*JH});
        }
      }
    }
  }

  return Uval_list;
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

  // Check lowest eigenvalues for each nelec sector
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

}