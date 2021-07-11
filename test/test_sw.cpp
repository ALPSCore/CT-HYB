#include <memory>
#include <tuple>
#include <iostream>
#include <alps/params.hpp>

# include "test_sw.hpp"
# include "../src/model/atomic_model.hpp"


TEST(SlidingWindow, copy) {
  using MODEL = REAL_EIGEN_BASIS_MODEL;
  double beta = 10.0;
  int nflavors = 1;
  int n_section = 10;
  auto p_model = std::shared_ptr<MODEL>(new MODEL(nflavors));
  auto sw = SlidingWindowManager<MODEL>(n_section, p_model, beta);

  SlidingWindowManager<MODEL> sw_copy(sw);
  ASSERT_TRUE(sw == sw_copy);

  SlidingWindowManager<MODEL> sw2(1, sw.get_p_model(), sw.get_beta(), sw.get_operators());
  sw2 = sw;
  ASSERT_TRUE(sw == sw2);

  SlidingWindowManager<MODEL> sw3(sw);
  ASSERT_TRUE(sw == sw3);

  SlidingWindowManager<MODEL> sw4;
  sw4 = sw;
  ASSERT_TRUE(sw == sw4);
}

TEST(SlidingWindow, uniform) {
  using MODEL = REAL_EIGEN_BASIS_MODEL;
  double beta = 10.0;
  int nflavors = 1;
  int n_section = 10;
  auto p_model = std::shared_ptr<MODEL>(new MODEL(nflavors));
  auto sw = SlidingWindowManager<MODEL>(n_section, p_model, beta);

  ASSERT_FLOAT_EQ(sw.get_tau_edge(0), 0.0);
  ASSERT_FLOAT_EQ(sw.get_tau_edge(n_section), beta);
  for (auto i=1; i<n_section; ++i) {
    ASSERT_NEAR((beta * i)/(n_section), sw.get_tau_edge(i), 1e-10);
  }

  // Double n_section
  auto n_section2 = 2 * n_section;
  sw.set_uniform_mesh(n_section2);
  ASSERT_FLOAT_EQ(sw.get_tau_edge(0), 0.0);
  ASSERT_FLOAT_EQ(sw.get_tau_edge(n_section2), beta);
  for (auto i=1; i<n_section2; ++i) {
    ASSERT_NEAR((beta * i)/(n_section2), sw.get_tau_edge(i), 1e-10);
  }
}

TEST(SlidingWindow, nonuniform) {
  using MODEL = REAL_EIGEN_BASIS_MODEL;
  double beta = 10.0;
  int nflavors = 1;
  auto p_model = std::shared_ptr<MODEL>(new MODEL(nflavors));

  std::vector<double> tau_edges {0.0, 0.1*beta, 0.5*beta, 0.8*beta, beta};
  auto sw = SlidingWindowManager<MODEL>(p_model, beta, tau_edges);

  for (int i=0; i<tau_edges.size(); ++i) {
    ASSERT_EQ(sw.get_tau(i), tau_edges[i]);
  }
}

TEST(SlidingWindow, move) {
  using MODEL = REAL_EIGEN_BASIS_MODEL;
  double beta = 10.1;
  int nflavors = 1;
  int n_section = 10;
  auto p_model = std::shared_ptr<MODEL>(new MODEL(nflavors));
  auto sw = SlidingWindowManager<MODEL>(n_section, p_model, beta);
  operator_container_t ops;

  ASSERT_EQ(n_section, sw.get_position_left_edge());
  ASSERT_EQ(0, sw.get_position_right_edge());

  // Move left edge to pos 2 (right edge stays at 0.)
  sw.move_left_edge_to(2);
  ASSERT_EQ(2, sw.get_position_left_edge());
  ASSERT_EQ(0, sw.get_position_right_edge());

  // Move left edge to pos 0 (right edge stays at 0.)
  sw.move_left_edge_to(0);
  ASSERT_EQ(0, sw.get_position_left_edge());
  ASSERT_EQ(0, sw.get_position_right_edge());

  // Move left edge to pos n_section, then right edge to the same position.
  // The window width will be 0.
  sw.move_left_edge_to(n_section);
  sw.move_right_edge_to(n_section);
  ASSERT_EQ(n_section, sw.get_position_left_edge());
  ASSERT_EQ(n_section, sw.get_position_right_edge());

  // Set the move direction to left, move the window twice
  sw.set_direction(ITIME_LEFT);
  sw.move_window_to_next_position();
  ASSERT_EQ(n_section-1, sw.get_position_left_edge());
  ASSERT_EQ(n_section-1, sw.get_position_right_edge());
  ASSERT_EQ(ITIME_RIGHT, sw.get_direction_move_local_window());

  sw.move_window_to_next_position();
  ASSERT_EQ(n_section-2, sw.get_position_left_edge());
  ASSERT_EQ(n_section-2, sw.get_position_right_edge());

  // Move right edge to pos 0, then left edge to pos 2.
  // Set the move direction to left, move the window once
  sw.move_right_edge_to(0);
  sw.move_left_edge_to(2);
  sw.set_direction(ITIME_RIGHT);
  sw.move_window_to_next_position();
  ASSERT_EQ(3, sw.get_position_left_edge());
  ASSERT_EQ(1, sw.get_position_right_edge());
  ASSERT_EQ(ITIME_LEFT, sw.get_direction_move_local_window());
}

/*
 * Single-orbital Hubbard atom at half filling
 */
TEST(SlidingWindow, trace) {
  using SCALAR = std::complex<double>;
  using MODEL = COMPLEX_EIGEN_BASIS_MODEL;

  double beta = 2.0;
  int n_section = 10;

  auto onsite_U = 2.0;
  auto mu = 0.5*onsite_U;
  auto nflavors = 2;

  std::vector<std::tuple<int, int, int, int, SCALAR> > Uval_list{ {0, 1, 1, 0, onsite_U} };
  std::vector<std::tuple<int, int, SCALAR> > t_list {{0,0,-mu}, {1,1,-mu}};

  auto p_model = std::shared_ptr<MODEL>(new MODEL(nflavors, t_list, Uval_list));

  // Check eigen states
  std::vector<int> nelec_sectors_ref = {2, 1, 1, 0};
  std::vector<double> min_enes_ref = {0, -0.5*onsite_U, -0.5*onsite_U, 0};
  std::vector<double> min_enes;
  std::vector<int> nelec_sectors;
  for (auto sector=0; sector<4; ++sector) {
      min_enes.push_back(p_model->min_energy(sector) + p_model->get_reference_energy());
      nelec_sectors.push_back(p_model->nelec_sector(sector));
  }
  ASSERT_EQ(min_enes_ref, min_enes);
  ASSERT_EQ(nelec_sectors_ref, nelec_sectors);
  ASSERT_EQ(p_model->get_reference_energy(), -0.5*onsite_U);

  auto up = 0, dn = 1;

  // Partition function
  EXTENDED_COMPLEX Z = SlidingWindowManager<MODEL>(n_section, p_model, beta).compute_trace();
  std::complex<double> Z_ref = p_model->compute_z(beta);
  ASSERT_NEAR(std::abs(mycast<std::complex<double>>(Z) - Z_ref), 0.0, 1e-8);

  // n_up(tau)
  for (auto tau : {0.0, 0.1*beta, 0.5*beta}) {
    operator_container_t ops;
    ops.insert(psi(OperatorTime(tau, 1), CREATION_OP, up));
    ops.insert(psi(OperatorTime(tau, 0), ANNIHILATION_OP, up));
    auto sw = SlidingWindowManager<MODEL>(n_section, p_model, beta, ops);
    for (auto right_pos=0; right_pos <= n_section; ++right_pos) {
      for (auto left_pos=right_pos; left_pos <= n_section; ++left_pos) {
        sw.move_edges_to(left_pos, right_pos);
        auto nup = mycast<std::complex<double>>(sw.compute_trace()/Z);
        ASSERT_NEAR(std::abs(nup - 0.5), 0.0, 1e-8);
      }
    }
  }

  
  // -G(tau) = Tr[e^{-(beta-tau) H} c_up e^{-tau H} c^dagger_up]/Z
  for (auto tau : {0.0*beta, 0.1*beta, 0.5*beta, beta}) {
    operator_container_t ops;
    ops.insert(psi(OperatorTime(0,   0), CREATION_OP, up));
    ops.insert(psi(OperatorTime(tau, 1), ANNIHILATION_OP, up));
    auto sw = SlidingWindowManager<MODEL>(n_section, p_model, beta, ops);
    auto gtau = mycast<double>(sw.compute_trace()/Z);
    auto gtau_ref = 0.5 * (
      std::exp(-0.5*onsite_U*tau)/(1+std::exp(-0.5*beta*onsite_U)) +
      std::exp(+0.5*onsite_U*tau)/(1+std::exp(+0.5*beta*onsite_U))
    );
    ASSERT_NEAR(gtau_ref, gtau, 1e-8);
  }
}