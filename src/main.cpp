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

#include "hdf5/boost_any.hpp"
#include "solver.hpp"

//void save_ouput_file(const alps::params &parameters, const alps::accumulators::result_set &mc_results, const std::map<std::string,boost::any> &results) {
  //std::string output_file = parameters["outputfile"];
  //alps::hdf5::archive ar(boost::filesystem::path(output_file), "w");
  //ar["/parameters"] << parameters;
  //ar["/simulation/results"] << mc_results;
//}

int main(int argc, const char *argv[]) {
  alps::params par(argc, argv);

  par.define<std::string>("algorithm", "complex-matrix", "Name of algorithm (real-matrix, complex-matrix)");

  char **argv_tmp = const_cast<char **>(argv);//FIXME: ugly solution
  alps::mpi::environment env(argc, argv_tmp);
  alps::mpi::communicator c;

  alps::accumulators::result_set results;

  //set up solver
  boost::shared_ptr<alps::cthyb::Solver> p_solver;
  if (par["algorithm"].as<std::string>() == "real-matrix") {
    alps::cthyb::MatrixSolver<double>::define_parameters(par);
    if (par.help_requested(std::cout)) { exit(0); } //If help message is requested, print it and exit normally.

    p_solver.reset(new alps::cthyb::MatrixSolver<double>(par));
  } else if (par["algorithm"].as<std::string>() == "complex-matrix") {
    alps::cthyb::MatrixSolver<std::complex<double> >::define_parameters(par);
    if (par.help_requested(std::cout)) { exit(0); } //If help message is requested, print it and exit normally.

    p_solver.reset(new alps::cthyb::MatrixSolver<std::complex<double> >(par));
  } else {
    throw std::runtime_error("Unknown algorithm: " + par["algorithm"].as<std::string>());
  }

  //solve the model
  p_solver->solve();

  //write the results into a hdf5 file
  if (c.rank() == 0) {
    std::string output_file = par["outputfile"];
    alps::hdf5::archive ar(boost::filesystem::path(output_file), "w");
    ar["/parameters"] << par;
    ar["/simulation/results"] << p_solver->get_accumulated_results();
    {
      const std::map<std::string,boost::any> &results = p_solver->get_results();
      for (std::map<std::string,boost::any>::const_iterator it = results.begin(); it != results.end(); ++it) {
        ar["/" + it->first] << it->second;
      }
    }
  }

  return 0;
}
