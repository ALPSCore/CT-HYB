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

#include "util.hpp"

#include <alps/utilities/signal.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>
#ifdef ALPS_HAVE_MPI
#include <alps/utilities/mpi.hpp>
#include "mc/mympiadapter.hpp"
#endif
#include "mc/mymcadapter.hpp"
#include "postprocess.hpp"

int global_mpi_rank;

template<class SOLVER_TYPE>
int run_simulation(int argc, const char *argv[], typename alps::parameters_type<SOLVER_TYPE>::type &parameters) {
#ifdef ALPS_HAVE_MPI
  typedef mymcmpiadapter<SOLVER_TYPE> sim_type;
#else
  typedef mymcadapter<SOLVER_TYPE> sim_type;
#endif

  SOLVER_TYPE::define_parameters(parameters);
  if (parameters.help_requested(std::cout)) {
    exit(0);
  }

  char **argv_tmp = const_cast<char **>(argv);//ugly solution
#ifdef ALPS_HAVE_MPI
  alps::mpi::environment env(argc, argv_tmp);
  alps::mpi::communicator c;
  c.barrier();
  global_mpi_rank = c.rank();
  if (global_mpi_rank == 0) {
    std::cout << "Creating simulation..." << std::endl;
  }
  sim_type sim(parameters, c);
  const boost::function<bool()> cb = alps::stop_callback(c, size_t(parameters["timelimit"]));
#else
  global_mpi_rank = 0;
  sim_type sim(parameters);
  const boost::function<bool()> cb = alps::stop_callback(size_t(parameters["timelimit"]));
#endif

  sim.run(cb);

  // Saving to the output file
#ifdef ALPS_HAVE_MPI
  if (c.rank() == 0) {
#endif
    typename alps::results_type<SOLVER_TYPE>::type results = alps::collect_results(sim);
    std::string output_file = parameters["outputfile"];
    alps::hdf5::archive ar(boost::filesystem::path(output_file), "w");
    ar["/parameters"] << parameters;
    ar["/simulation/results"] << results;
    compute_greens_functions<SOLVER_TYPE>(results, parameters, ar);
    if (parameters["N_LEGENDRE_N2_MEASUREMENT"] > 0) {
      N2_correlation_function<SOLVER_TYPE>(results, parameters, ar, global_mpi_rank == 0);
    }
    if (global_mpi_rank == 0) {
      show_statistics<SOLVER_TYPE>(results, parameters, ar);
    }
#ifdef ALPS_HAVE_MPI
  } else {
    alps::collect_results(sim);
  }
#endif

#ifdef ALPS_HAVE_MPI
  c.barrier();
#endif

  return 0;
}
