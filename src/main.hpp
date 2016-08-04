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

  std::pair<bool, bool> r = sim.run(cb);

  // Saving to the output file
#ifdef ALPS_HAVE_MPI
  if (c.rank() == 0) {
#endif
    if (!r.second) {
      throw std::runtime_error("Master process is not thermalized yet. Increase simulation time!");
    }
    typename alps::results_type<SOLVER_TYPE>::type results = alps::collect_results(sim);
    std::string output_file = parameters["outputfile"];
    alps::hdf5::archive ar(boost::filesystem::path(output_file), "w");
    ar["/parameters"] << parameters;
    ar["/simulation/results"] << results;

    {
      compute_greens_functions<SOLVER_TYPE>(results, parameters, ar);
      compute_G1<SOLVER_TYPE>(results, parameters, sim.get_rotmat_Delta(), ar, global_mpi_rank == 0);
      if (parameters["measurement.two_time_G2.on"] != 0) {
        compute_two_time_G2<SOLVER_TYPE>(results, parameters, sim.get_rotmat_Delta(), ar, global_mpi_rank == 0);
      }
      compute_euqal_time_G1<SOLVER_TYPE>(results, parameters, sim.get_rotmat_Delta(), ar, global_mpi_rank == 0);
      if (parameters["measurement.equal_time_G2.on"] != 0) {
        compute_euqal_time_G2<SOLVER_TYPE>(results, parameters, sim.get_rotmat_Delta(), ar, global_mpi_rank == 0);
      }
      compute_fidelity_susceptibility<SOLVER_TYPE>(results, parameters, ar);
      if (global_mpi_rank == 0) {
        sim.show_statistics(results);
      }
    }
#ifdef ALPS_HAVE_MPI
  } else {
    if (r.second) {
      alps::collect_results(sim);
    } else {
      throw std::runtime_error((boost::format(
          "Warning: MPI process %1% is not thermalized yet. Increase simulation time!") % global_mpi_rank).str());
    }
  }
#endif

#ifdef ALPS_HAVE_MPI
  c.barrier();
#endif

  return 0;
}
