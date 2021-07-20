#pragma once

#include <tuple>
#include <alps/params.hpp>

#include "../src/model/atomic_model.hpp"
#include "../src/model/hybridization_function.hpp"

#include <gtest.h>

// Check all eigen energies by applying H to an eigenstate.
template<typename SCALAR>
void
check_eigenes(const AtomicModelEigenBasis<SCALAR> &model) {
  using dense_matrix_t = Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>;
  using braket_t = typename AtomicModelEigenBasis<SCALAR>::BRAKET_T;

  for (auto sector=0; sector<model.num_sectors(); ++sector) {
    int dim_sector = model.dim_sector(sector);
    for (auto state=0; state<dim_sector; ++state) {
      dense_matrix_t evec(dim_sector, 1);
      evec.setZero();
      evec(state, 0) = 1.0; // sector-th eigenbasis
      auto eigenket = braket_t(sector, evec);
      auto eigenbra = braket_t(sector, evec.transpose());

      auto ene_from_ket = 0.0, ene_from_bra = 0.0;

      // hopping term
      for (const auto& tij: model.get_nonzero_t_vals()) {
        auto ket = braket_t(eigenket);
        model.apply_op_hyb_ket(ANNIHILATION_OP, std::get<1>(tij), ket);
        model.apply_op_hyb_ket(CREATION_OP, std::get<0>(tij), ket);
        ene_from_ket += static_cast<double>(get_real(std::get<2>(tij) * model.product(eigenbra, ket)));

        auto bra = braket_t(eigenbra);
        model.apply_op_hyb_bra(CREATION_OP, std::get<0>(tij), bra);
        model.apply_op_hyb_bra(ANNIHILATION_OP, std::get<1>(tij), bra);
        ene_from_bra += static_cast<double>(get_real(std::get<2>(tij) * model.product(bra, eigenket)));
      }

      // Coulomb interaction
      for (const auto& Uijkl: model.get_nonzero_U_vals()) {
        {
          auto ket = braket_t(eigenket);
          model.apply_op_hyb_ket(ANNIHILATION_OP, std::get<3>(Uijkl), ket);
          model.apply_op_hyb_ket(ANNIHILATION_OP, std::get<2>(Uijkl), ket);
          model.apply_op_hyb_ket(CREATION_OP,     std::get<1>(Uijkl), ket);
          model.apply_op_hyb_ket(CREATION_OP,     std::get<0>(Uijkl), ket);
          ene_from_ket += static_cast<double>(
              get_real(std::get<4>(Uijkl) * model.product(eigenbra, ket))
          );
        }

        {
          auto bra = braket_t(eigenbra);
          model.apply_op_hyb_bra(CREATION_OP,     std::get<0>(Uijkl), bra);
          model.apply_op_hyb_bra(CREATION_OP,     std::get<1>(Uijkl), bra);
          model.apply_op_hyb_bra(ANNIHILATION_OP, std::get<2>(Uijkl), bra);
          model.apply_op_hyb_bra(ANNIHILATION_OP, std::get<3>(Uijkl), bra);
          ene_from_bra += static_cast<double>(
              get_real(std::get<4>(Uijkl) * model.product(bra, eigenket))
          );
        }
      }

      ASSERT_NEAR(ene_from_ket, model.get_eigenvals_sector()[sector][state] + model.get_reference_energy(), 1e-8);
      ASSERT_NEAR(ene_from_bra, model.get_eigenvals_sector()[sector][state] + model.get_reference_energy(), 1e-8);
    }
  }

}

// Check sector_propagate_ket and sector_propagate_bra
template<typename SCALAR>
void
check_sector_propagate(const AtomicModelEigenBasis<SCALAR> &model) {
  using dense_matrix_t = Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>;
  using braket_t = typename AtomicModelEigenBasis<SCALAR>::BRAKET_T;

  for (auto sector=0; sector<model.num_sectors(); ++sector) {
    int dim_sector = model.dim_sector(sector);
    for (auto state=0; state<dim_sector; ++state) {
      dense_matrix_t evec(dim_sector, 1);
      evec.setZero();
      evec(state, 0) = 1.0; // sector-th eigenbasis
      auto eigenket = braket_t(sector, evec);
      auto eigenbra = braket_t(sector, evec.transpose());

      double eigen_ene = model.get_eigenvals_sector()[sector][state];

      double dt;
      if (std::abs(eigen_ene) < 1.0) {
        dt = 1.0;
      } else {
        dt = 1/eigen_ene;
      }

      auto eigenket_prop = braket_t(eigenket);
      model.sector_propagate_ket(eigenket_prop, dt);

      ASSERT_TRUE(
        std::abs(
          mycast<std::complex<double>>(
            model.product(eigenbra, eigenket_prop)/std::exp(-dt * eigen_ene)
          ) - 1.0
        ) < 1e-8
      );

      auto eigenbra_prop = braket_t(eigenbra);
      model.sector_propagate_bra(eigenbra_prop, dt);

      ASSERT_TRUE(
        std::abs(
          mycast<std::complex<double>>(
            model.product(eigenbra_prop, eigenket)/std::exp(-dt * eigen_ene)
          ) - 1.0
        ) < 1e-8
      );

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

// Test data F_ij(tau) = (i+1) * (j+1) * exp(-tau*beta) 
template<typename SCALAR>
SCALAR test_F_two_flavor(double tau, double beta, int i, int j) {
  check_true(i <= 1);
  check_true(j <= 1);
  std::complex<double> z;
  auto x = 2*(tau-0.5*beta)/beta;
  if (i == j) {
    z = (i+1) * (x + x*x);
  } else if (i==0 && j == 1) {
    z = std::complex<double>(1.0, +0.1) * (x + x*x);
  } else {
    z = std::complex<double>(1.0, -0.1) * (x + x*x);
  }
  return mycast<SCALAR>(z);
}

template<typename SCALAR>
void test_read_intpl_hyb_two_flavor() {
    auto ntau = 1000;
    auto beta = 2.0;
    auto nflavors = 2;
    auto Np1 = ntau + 1;
    std::string fname = "delta.txt";

    {
      std::ofstream of(fname.c_str());
      for (auto time = 0; time < Np1; time++) {
        auto tau = beta * time / (1. * ntau);
        for (auto i = 0; i < nflavors; i++) {
          for (auto j = 0; j < nflavors; j++) {
            std::complex<double> delta = - test_F_two_flavor<SCALAR>(beta-tau, beta, j, i);
            of << time << " " << i << " " << j << " " 
                << delta.real() << " " << delta.imag() << std::endl;;
          }
        }
      }
    }
    
    auto hyb = HybridizationFunction<SCALAR>(beta, fname, ntau, nflavors);

    // 0 < tau < beta
    auto atol = 1e-3;
    std::vector<double> tau0_list {0.0, beta, 2*beta, -beta};
    std::vector<double> sign_list {1.0, -1.0,    1.0,  -1.0};
    for (auto time = 1; time < 2*ntau; time++) {
      auto tau = beta * time/(2. * ntau);
      for (auto i = 0; i < nflavors; i++) {
        for (auto j = 0; j < nflavors; j++) {
          for (auto ishift=0; ishift < tau0_list.size(); ++ishift) {
            auto tau0 = tau0_list[ishift];
            auto sign = sign_list[ishift];
            ASSERT_NEAR(
              std::abs(
                hyb(tau+tau0, i, j) - sign * test_F_two_flavor<SCALAR>(tau, beta ,i, j)
              ), 0.0, atol
            );
          }
        }
      }
    }
}