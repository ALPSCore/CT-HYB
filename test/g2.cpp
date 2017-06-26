#include <boost/random.hpp>
#include "../src/measurement/g2.hpp"

#include <gtest.h>

TEST(G2, MeasureByHyb) {
  boost::random::mt19937 gen(100);
  boost::uniform_real<> uni_dist(0, 1);

  using SCALAR = double;

  int k = 50;
  int num_flavors = 4;

  double beta = 1.0;
  double Lambda = 1.0;
  int dim = 4;

  boost::shared_ptr<OrthogonalBasis> p_basis_f(new FermionicIRBasis(Lambda, dim));
  boost::shared_ptr<OrthogonalBasis> p_basis_b(new BosonicIRBasis(Lambda, dim));

  std::vector<psi> creation_ops;
  std::vector<psi> annihilation_ops;
  Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic> M(k, k);

  for (int i=0; i<k; ++i) {
    creation_ops.push_back(psi(beta*uni_dist(gen), CREATION_OP, static_cast<int>(num_flavors*uni_dist(gen))));
    annihilation_ops.push_back(psi(beta*uni_dist(gen), ANNIHILATION_OP, static_cast<int>(num_flavors*uni_dist(gen))));
  }

  for (int j=0; j<k; ++j) {
    for (int i=0; i<k; ++i) {
      M(i,j) = uni_dist(gen);
    }
  }

  Eigen::Tensor<SCALAR,7> r = measure_g2(beta, num_flavors, p_basis_f, p_basis_b, creation_ops, annihilation_ops, M);
  Eigen::Tensor<SCALAR,7> r_ref = measure_g2_ref(beta, num_flavors, p_basis_f, p_basis_b, creation_ops, annihilation_ops, M);

  Eigen::Tensor<SCALAR,7> rdiff = (r - r_ref).abs();
  Eigen::Tensor<SCALAR,0> diff = rdiff.sum();
  ASSERT_TRUE(diff.coeff() < 1E-10);

}
