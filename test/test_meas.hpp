#pragma once

#include <gtest.h>
#include <fstream>

#include <boost/shared_ptr.hpp>

#include <alps/accumulators.hpp>
#include <alps/mc/random01.hpp>
#include <alps/accumulators.hpp>

#include "../src/model/hybridization_function.hpp"
#include "../src/moves/mc_config.hpp"
#include "../src/moves/worm.hpp"
#include "../src/sliding_window/sliding_window.hpp"
#include "../src/measurement/all.hpp"

template<typename SCALAR>
void test_equal_time_G1(){
  using MODEL_T = AtomicModelEigenBasis<SCALAR>;
  using SW_T = SlidingWindowManager<AtomicModelEigenBasis<SCALAR>>;

  // Single-orbital Hubbard atom (filled)
  auto onsite_U = 2.0;
  auto mu = - 20.0;
  auto beta = 5.0;
  auto nflavors = 2;
  auto ntau = 2;
  auto n_section_sw = 10;

  auto up = 0;
  auto dn = 1;
  auto num_ins = 2;

  std::vector<std::tuple<int, int, int, int, SCALAR> > Uval_list{{0, 1, 1, 0, onsite_U}};
  std::vector<std::tuple<int, int, SCALAR> > t_list {{0, 0, -mu}, {1, 1, -mu}};
  auto p_model = std::shared_ptr<MODEL_T>(new MODEL_T(nflavors, t_list, Uval_list));

  // Zero hybridization function
  boost::multi_array<SCALAR,3> F_data(boost::extents[nflavors][nflavors][ntau+1]);
  std::fill(F_data.origin(), F_data.origin()+F_data.num_elements(), 0.0);
  auto p_F = std::shared_ptr<HybridizationFunction<SCALAR>>(
    new HybridizationFunction<SCALAR>(beta, ntau, nflavors, F_data)
  );

  auto p_worm = std::shared_ptr<EqualTimeGWorm<1>>(new EqualTimeGWorm<1>());
  p_worm->set_time(         0, 0.5*beta);
  p_worm->set_flavor(       0, up); // c_op
  p_worm->set_flavor(       1, up); // cdagg_op
  auto worm_ops_ =  p_worm->get_operators();
  operator_container_t worm_ops(worm_ops_.begin(), worm_ops_.end());

  auto mc_config = MonteCarloConfiguration<SCALAR>(p_F);
  mc_config.set_worm(p_worm);

  auto sw = SW_T(n_section_sw, p_model, beta, worm_ops);

  alps::random01 rng;
  EqualTimeG1Meas<SCALAR,SW_T> worm_meas(&rng, beta, nflavors, num_ins);
  alps::accumulators::accumulator_set measurements;
  worm_meas.create_alps_observable(measurements);
  worm_meas.measure(mc_config, sw, measurements);

  alps::accumulators::result_set results(measurements);

  std::map<std::string,boost::any> ar;
  double sign = 1.0;
  double worm_space_vol_rat = beta * nflavors;// G is diagonal in spin.
  compute_equal_time_G1(results, nflavors, beta, sign, worm_space_vol_rat, ar);
  auto res = boost::any_cast<boost::multi_array<std::complex<double>,2>&>(ar["EQUAL_TIME_G1"]);
  for (auto f0=0; f0<2; ++f0) {
    for (auto f1=0; f1<2; ++f1) {
      if (f0==f1) {
        ASSERT_TRUE(std::abs(res[f0][f1] - 1.0) < 1e-8/beta);
      } else {
        ASSERT_TRUE(std::abs(res[f0][f1]) < 1e-8/(beta*beta));
      }
    }
  }

}
