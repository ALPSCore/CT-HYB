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

#include <fstream>
#include <cstdio>
#include <boost/filesystem/operations.hpp>
#include <alps/utilities/fs/remove_extensions.hpp>
#include <alps/utilities/fs/get_dirname.hpp>
#include <alps/utilities/fs/get_basename.hpp>

#include "hdf5/boost_any.hpp"
#include "solver.hpp"
#include "common/logger.hpp"
#include "moves/worm.hpp"
#include "measurement/all.hpp"
#include "postprocess.hpp"




int main(int argc, const char *argv[]) {
  //Here we construct a parameter object by parsing an ini file.
  alps::params par(argc, argv);

  par.define<std::string>("algorithm", "complex-matrix", "Name of algorithm (real-matrix, complex-matrix)");
  par.define<std::string>("target_worm_space_name", "", "Name of a worm space to be simulated. Leave it blank to simulate all worm spaces needed.");

  char **argv_tmp = const_cast<char **>(argv);//FIXME: ugly solution
  alps::mpi::environment env(argc, argv_tmp);

  //set up solver
  std::shared_ptr<alps::cthyb::Solver> p_solver;
  if (par["algorithm"].as<std::string>() == "real-matrix") {
    alps::cthyb::MatrixSolver<double>::define_parameters(par);
    if (par.help_requested(std::cout)) { exit(0); } //If help message is requested, print it and exit normally.
  } else if (par["algorithm"].as<std::string>() == "complex-matrix") {
    alps::cthyb::MatrixSolver<std::complex<double> >::define_parameters(par);
    if (par.help_requested(std::cout)) { exit(0); } //If help message is requested, print it and exit normally.
  } else {
    throw std::runtime_error("Unknown algorithm: " + par["algorithm"].as<std::string>());
  }

  std::vector<ConfigSpaceEnum::Type> defined_worm_spaces;
  if (par["algorithm"].as<std::string>() == "real-matrix") {
    defined_worm_spaces = alps::cthyb::MatrixSolver<double>::get_defined_worm_spaces(par);
  } else if (par["algorithm"].as<std::string>() == "complex-matrix") {
    defined_worm_spaces = alps::cthyb::MatrixSolver<std::complex<double>>::get_defined_worm_spaces(par);
  }

  // Split MPI communicator
  int num_defined_worm_spaces = defined_worm_spaces.size();
  auto comm = alps::mpi::communicator();
  if (comm.rank() == 0) {
    std::cout << "Creating simulation for " << std::to_string(num_defined_worm_spaces) <<
      " worm spaces with " << std::to_string(comm.size()) << " MPI processes..." << std::endl;
  }
  auto rank = comm.rank();
  if (comm.size() < num_defined_worm_spaces) {
    throw std::runtime_error(
      std::string("Number of worm spaces ") + std::to_string(num_defined_worm_spaces) + 
      std::string(" is larger than number of MPI processes!"));
  }
  int subgrp = rank%num_defined_worm_spaces;
  alps::mpi::communicator sub_comm;
  {
    MPI_Comm sub_comm_;
    MPI_Comm_split(MPI_COMM_WORLD, subgrp, 0, &sub_comm_);
    sub_comm = alps::mpi::communicator(sub_comm_, alps::mpi::take_ownership);
  }
  par["target_worm_space"] = subgrp;
  auto target_worm_space = defined_worm_spaces[subgrp];
  std::string prefix_global = alps::fs::remove_extensions(par["outputfile"]);
  auto prefix = alps::fs::remove_extensions(par["outputfile"]) + "_wormspace_"
    + ConfigSpaceEnum::to_string(target_worm_space);
  const std::string outputfile = prefix + ".out.h5";
  par["outputfile"] = outputfile;
  std::cout << "Assigning rank " << std::to_string(rank)
    << " to subgroup " << std::to_string(subgrp) << " for "
    << ConfigSpaceEnum::to_string(target_worm_space) << std::endl;

  if (par["algorithm"].as<std::string>() == "real-matrix") {
    p_solver.reset(new alps::cthyb::MatrixSolver<double>(par, sub_comm));
  } else if (par["algorithm"].as<std::string>() == "complex-matrix") {
    p_solver.reset(new alps::cthyb::MatrixSolver<std::complex<double> >(par, sub_comm));
  }

  logger_out = std::ofstream("log_" + prefix + "_rank" + std::to_string(sub_comm.rank()) + ".txt");

  // Remove the exisiting old output file if any
  if (sub_comm.rank() == 0 && file_exists(outputfile)) {
    logger_out << "Removing the old output file " << outputfile << "..." << std::endl;
    std::remove(outputfile.c_str());
  }

  // Remove the exising output directory if any
  /*
  if (comm.rank()==0) {
    std::string dirname = prefix_global + "_results";
    std::cout << "Preparing output dir " << dirname << "..." << std::endl;
    if (boost::filesystem::exists(dirname)) {
      if (!boost::filesystem::is_directory(dirname)) {
        throw std::runtime_error("Please remove " + dirname + "!");
      }
      //boost::filesystem::remove_all(dirname);
    } else {
      boost::filesystem::create_directory(dirname);
    }
  }
  */

  //solve the model
  p_solver->solve();

  //write the results into a hdf5 file
  sub_comm.barrier();
  if (sub_comm.rank() == 0) {
    logger_out << "Writing result into " << prefix + ".out.h5...";
    alps::hdf5::archive ar(prefix + ".out.h5", "a");
    ar["/parameters"] << par;
    ar["/simulation/results"] << p_solver->get_accumulated_results();
    {
      const std::map<std::string,boost::any> &results = p_solver->get_results();
      for (std::map<std::string,boost::any>::const_iterator it = results.begin(); it != results.end(); ++it) {
        ar["/" + it->first] << it->second;
      }
    }
    logger_out << "Done!" << std::endl;
  }
  logger_out.flush();
  sub_comm.barrier();

  if (comm.rank() == 0) {
    alps::hdf5::archive ar(prefix_global + ".out.h5", "w");
    ar["/parameters"] << par;
    for (auto ws: defined_worm_spaces) {
      ar["/worm_spaces/" + ConfigSpaceEnum::to_string(ws)] << 1;
    }
  }

  return 0;
}
