/************************************************************************************
 *
 * Hybridization expansion code for multi-orbital systems with general interactions
 *
 * Copyright (C) 2016 by Hiroshi Shinaoka <h.shinaoka@gmail.com>,
 *                                    Emanuel Gull <egull@umich.edu>,
 *                                    Philipp Werner <philipp.werner@unifr.ch>
 *
 *
 * This software is published under the GNU General Public License version 2.
 * See the file LICENSE.txt.
 *
 *************************************************************************************/

#include "main.hpp"
#include "impurity.hpp"

int main(int argc, const char* argv[]) {
  typedef HybridizationSimulation<ImpurityModelEigenBasis<double> > REAL_MATRIX_SOLVER;
  typedef HybridizationSimulation<ImpurityModelEigenBasis<std::complex<double> > > COMPLEX_MATRIX_SOLVER;

  alps::params par(argc, argv);
  par.define<std::string>("ALGORITHM", "complex-matrix", "Name of algorithm (real-matrix, complex-matrix)");

  if (par["ALGORITHM"].as<std::string>()=="real-matrix") {
    std::cout << "Calling real-matrix solver..." << std::endl;
    return run_simulation<REAL_MATRIX_SOLVER>(argc, argv, par);
  } else if (par["ALGORITHM"].as<std::string>()=="complex-matrix") {
    std::cout << "Calling complex-matrix solver..." << std::endl;
    return run_simulation<COMPLEX_MATRIX_SOLVER>(argc, argv, par);
  } else {
    throw std::runtime_error("Unknown algorithm: "+par["algorithm"].as<std::string>());
  }
}
