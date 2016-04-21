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
#pragma once

#include <alps/utilities/signal.hpp>
#include <alps/utilities/mpi.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/mpiadapter.hpp>
#include <alps/mc/stop_callback.hpp>

#include "mympiadapter.hpp"
#include "postprocess.hpp"

#undef BUILD_PYTHON_MODULE

//bool stop_callback(boost::posix_time::ptime const & end_time) {
//stops the simulation if time > end_time or if signals received.
  //static alps::signal signal;
  //return !signal.empty() || boost::posix_time::second_clock::local_time() > end_time;
///}
int global_mpi_rank;

template<class SOLVER_TYPE>
int run_simulation(int argc, const char* argv[], typename alps::parameters_type<SOLVER_TYPE>::type& parameters) {
  typedef mymcmpiadapter<SOLVER_TYPE> sim_type;

  SOLVER_TYPE::define_parameters(parameters);
  if (parameters.help_requested(std::cout)) {
    exit(0);
  }

  char** argv_tmp = const_cast<char**>(argv);//ugly solution
  alps::mpi::environment env(argc, argv_tmp);

  alps::mpi::communicator c;
  c.barrier();
  global_mpi_rank=c.rank();
  if (global_mpi_rank==0) {
    std::cout << "Creating simulation..." << std::endl;
  }
  sim_type sim(parameters, c);

  // Run the simulation
  sim.run(alps::stop_callback(size_t(parameters["timelimit"])));

  // Saving to the output file
  if (global_mpi_rank==0){
    typename alps::results_type<SOLVER_TYPE>::type results = alps::collect_results(sim);
    std::string output_file = parameters["outputfile"];
    alps::hdf5::archive ar(boost::filesystem::path(output_file), "w");
    ar["/parameters"] << parameters;
    ar["/simulation/results"] << results;
    compute_greens_functions<SOLVER_TYPE>(results, parameters, ar);
  } else{
    alps::collect_results(sim);
  }
  c.barrier();

  return 0;
}
